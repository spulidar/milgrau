"""Sequential SCC batch submissions with explicit destructive-replacement opt-in.

Every staged package is preflighted offline; invalid IDs are excluded from live SCC requests.
A timeout, unknown remote outcome or per-ID journal conflict stops by default;
never retry a POST or DELETE automatically, and never remove staging files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from milgrau.scc.packages import validate_upload_package
from milgrau.scc.state import BLOCKED_RESTART, SubmissionStore, state_root
from milgrau.scc.workflow import submit


@dataclass
class BatchReport:
    total: int
    submitted: int = 0
    replaced: int = 0
    identical: int = 0
    skipped_existing: int = 0
    failed: int = 0
    invalid: int = 0
    stopped_at: str | None = None
    error: str | None = None

    @property
    def remaining(self) -> int:
        return self.total - (
            self.submitted + self.replaced + self.identical
            + self.skipped_existing + self.failed
        )

    @property
    def successful(self) -> bool:
        return self.failed == 0 and self.remaining == 0


def submit_batch(
    config: Mapping[str, Any],
    packages: Sequence[Path],
    client: Any,
    *,
    replace_existing: bool = False,
    continue_on_error: bool = False,
    report_line: Callable[[str], None] = print,
) -> BatchReport:
    """Batch one at a time; never treat an existing SCC ID as upload ownership.

    Default: submit absent valid IDs and skip remote-existing IDs without modifying
    their local journal. With replace_existing, invoke the same exact-ID,
    backup-before-delete workflow as the single-upload command. Invalid IDs
    are always skipped; optional continue_on_error proceeds to independent IDs
    after remote errors without clearing their durable blocked journal.
    """
    packages = sorted(packages, key=lambda path: path.name)
    report = BatchReport(total=len(packages))
    if not packages:
        report_line("No local pending SCC packets.")
        return report

    # Validate every package without turning an unrelated bad cache entry
    # into a global blocker. Do NOT issue remote requests for invalid IDs.
    manifest_by_id: dict[str, dict[str, Any]] = {}
    invalid_ids: set[str] = set()
    for path in packages:
        try:
            manifest = validate_upload_package(path)
            identity = str(manifest["measurement_id"])
            if identity in manifest_by_id:
                raise ValueError("Duplicate staged SCC Measurement_ID.")
            manifest_by_id[identity] = manifest
        except Exception as exc:
            report_line(
                f"PRECHECK INVALID {path.name}: {type(exc).__name__}: {exc}; "
                "this ID will not be uploaded."
            )
            invalid_ids.add(path.name)
            report.failed += 1
            report.invalid += 1
    report_line(
        f"Preflight: {len(packages) - report.invalid}/{len(packages)} "
        "local packages valid; invalid IDs will be skipped."
    )
    root = state_root(config)
    for index, path in enumerate(packages, start=1):
        identity = path.name
        prefix = f"[{index}/{len(packages)}] {identity}"
        if identity in invalid_ids:
            continue
        try:
            with SubmissionStore(root) as store:
                journal = store.get(identity)
            # Existing remote measurements are a safe read-only no-op when
            # no replacement was requested, regardless of old journal state.
            # Querying SCC cannot reset an uncertain operation.
            existing = client.measurement(identity)
            if existing is not None and not replace_existing:
                report.skipped_existing += 1
                report_line(f"{prefix}: SKIPPED — already exists on SCC (unchanged).")
                continue
            if journal is not None and journal["state"] in BLOCKED_RESTART:
                raise RuntimeError(
                    f"Local SCC journal state={journal['state']} needs "
                    "single-ID reconciliation before any POST/DELETE."
                )
            if existing is None and journal is not None and journal["state"] in {
                "accepted", "processing", "completed", "downloaded", "cleaned",
            }:
                raise RuntimeError(
                    "This ID is recorded as previously submitted, but is now "
                    "absent on SCC. Reconcile the remote deletion before retry."
                )
            # submit() re-queries the API and independently enforces all
            # replacement preconditions, snapshots and exact-ID locks.
            is_replacement = bool(existing is not None and replace_existing)
            result = submit(
                config, path, client, replace_existing=is_replacement,
            )
            if result.get("status") == "already_identical":
                report.identical += 1
                report_line(f"{prefix}: IDENTICAL — remote measurement preserved.")
            elif is_replacement:
                report.replaced += 1
                report_line(f"{prefix}: REPLACED — same Measurement_ID; prior SCC archived.")
            else:
                report.submitted += 1
                report_line(f"{prefix}: SUBMITTED — remote API accepted.")
        except Exception as exc:
            report.failed += 1
            report.stopped_at = identity
            report.error = str(exc)
            report_line(
                f"{prefix}: STOPPED — {type(exc).__name__}: {exc}. "
                "No further IDs will be processed; inspect status/journal first."
            )
            if not continue_on_error:
                break
            report_line(
                f"{prefix}: other IDs may continue under --continue-on-error; "
                "this ID remains blocked."
            )
    return report
