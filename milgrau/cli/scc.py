"""SCC CLI: explicit upload, read-only status and retained result downloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from milgrau.config.loader import load_config
from milgrau.scc.packages import discover_pending, validate_upload_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="milgrau-scc", description="Optional native SCC upload workflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    upload = sub.add_parser("upload", help="Validate or explicitly submit an SCC packet.")
    upload.add_argument("--id", dest="identity", help="Exact 15-character SCC Measurement_ID.")
    upload.add_argument(
        "--replace-existing", action="store_true",
        help="Replace remote-existing IDs only with --confirm (no mandatory backup). "
             "With no --id, replaces eligible existing IDs in batch.",
    )
    upload.add_argument(
        "--resume-after-delete", action="store_true",
        help="Manual recovery for an uncertain DELETE only: --id ID --confirm, "
             "original verified archive or same-revision waiver, confirmed remote absence. "
             "Sends one POST, never repeats DELETE.",
    )
    action = upload.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run", action="store_true",
        help="Validate pending packages locally; never connect to the SCC.",
    )
    action.add_argument(
        "--confirm", action="store_true",
        help="Explicitly authorize upload, or same-ID DELETE and replacement "
             "with --replace-existing (no mandatory backup).",
    )

    status = sub.add_parser("status", help="List local pending packets or query remote SCC.")
    status.add_argument("--id", dest="identity", help="Filter by Measurement_ID.")
    status.add_argument(
        "--remote", action="store_true", help="Query remote SCC for one ID or all locally known IDs (credentials required).",
    )
    status.add_argument(
        "--backup", action="store_true",
        help="Read-only download and verification of existing SCC remote inputs/results; requires --remote --id.",
    )
    download = sub.add_parser(
        "download", help="Read-only SCC result download into scientific session/scc/scc-results."
    )
    download.add_argument(
        "--id", dest="identity",
        help="Exact SCC Measurement_ID; omit to download every completed locally known SCC ID.",
    )
    download.add_argument(
        "--session", help="Scientific Session_ID for a remote record not uploaded by MILGRAU."
    )
    return parser


def _client(config: dict[str, Any]):
    """Lazy-load HTTP extras only for explicit remote operations."""
    from milgrau.scc.client import SCCClient, SCCCredentials

    catalog = config.get("_scc_catalog", {})
    server = catalog.get("server", {})
    if not isinstance(server, dict) or not server.get("base_url"):
        raise ValueError("Set scc.yaml server.base_url before remote SCC operations.")
    http = config["scc"].get("http", {})
    return SCCClient(
        server["base_url"],
        SCCCredentials.from_environment(),
        timeout=float(http.get("timeout_seconds", 60)),
        get_retries=http.get("get_retries", 2),
    )


def _upload(config: dict[str, Any], args: argparse.Namespace) -> int:
    # Reject contradictory/destructive mode flags before reading staging,
    # credentials, journals, or performing any remote API operation.
    if args.resume_after_delete and (
        not args.confirm or not args.identity or args.replace_existing
    ):
        raise ValueError(
            "--resume-after-delete requires exactly --id ID --confirm; "
            "never combine it with batch mode or --replace-existing."
        )
    if args.replace_existing and args.dry_run:
        raise ValueError(
            "--replace-existing must be used with --confirm; "
            "offline --dry-run does not inspect remote SCC IDs."
        )
    packages = discover_pending(config, args.identity)
    if not packages:
        print("No SCC packages are staged in scientific session/scc/scc-inputs/.")
        return 0 if args.dry_run else 1
    if args.confirm and args.identity and len(packages) != 1:
        raise ValueError("Single-ID upload requires exactly one local pending package.")
    errors = 0
    if args.dry_run:
        for path in packages:
            try:
                manifest = validate_upload_package(path)
                print(
                    f"OK {path.name}: configuration={manifest['scc_configuration_id']}, "
                    f"products={manifest['scc_product_ids']}, "
                    f"files={', '.join(manifest['files'])}"
                )
            except Exception as exc:
                errors += 1
                print(f"INVALID {path.name}: {exc}", file=sys.stderr)
        print(f"Dry-run: {len(packages)-errors}/{len(packages)} valid; no network request.")
        return 1 if errors else 0

    if args.identity is None:
        from milgrau.scc.batch import submit_batch

        report = submit_batch(
            config, [Path(path) for path in packages], _client(config),
            replace_existing=args.replace_existing,
        )
        print(
            "Batch summary: "
            f"total={report.total}; submitted={report.submitted}; "
            f"replaced={report.replaced}; identical={report.identical}; "
            f"skipped_existing={report.skipped_existing}; "
            f"failed={report.failed}; invalid={report.invalid}; "
            f"unprocessed={report.remaining}."
        )
        if report.stopped_at:
            print(
                f"Batch encountered an error at {report.stopped_at}. "
                "Review the SCC status and durable journal for that ID."
            )
        print("SCC inputs, scientific data and durable SCC archives were retained.")
        return 0 if report.successful else 1

    # Single-ID operation uses the same guarded workflow as before.
    path = Path(packages[0])
    validate_upload_package(path)
    from milgrau.scc.workflow import resume_after_uncertain_delete, submit

    client = _client(config)
    if args.resume_after_delete:
        result = resume_after_uncertain_delete(config, path, client)
    else:
        result = submit(
            config, path, client, replace_existing=args.replace_existing,
            skip_backup=args.replace_existing,
        )
    print(
        f"SCC API confirmed {result['measurement_id']}: {result['remote_page']}. "
        "Submission is journalled in .state/scc; session SCC inputs are "
        "retained until the operator manages them manually."
    )
    return 0


def _status(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.backup and (not args.remote or not args.identity):
        raise ValueError("Remote archive --backup requires --id ID --remote.")
    # Remote measurements may outlive manually retained session SCC inputs.
    packages = discover_pending(config)
    if args.identity is not None:
        packages = [path for path in packages if path.name == args.identity]
    local_errors = 0
    for path in packages:
        try:
            manifest = validate_upload_package(path)
            print(
                f"pending {path.name} ({manifest['solar_regime']} / "
                f"{manifest['scc_configuration_id']}); local integrity OK"
            )
        except Exception as exc:
            local_errors += 1
            print(f"invalid pending package {path.name}: {exc}", file=sys.stderr)
    # Remote status is read-only and must remain available even if local
    # staged files are corrupted or were manually renamed incompletely.
    if not packages:
        print("No matching local SCC pending packets.")
    from milgrau.scc.state import SubmissionStore, state_root
    from milgrau.scc.workflow import reconciliation_guidance, reconcile_status

    with SubmissionStore(state_root(config)) as store:
        recorded = (
            [row] if args.identity and (row := store.get(args.identity)) is not None
            else ([] if args.identity else store.list())
        )
        for row in recorded:
            print(
                f"journal {row['measurement_id']}: state={row['state']}; "
                f"session={row['scientific_session_id']}; "
                f"archive={row['archive_path'] or '-'}"
            )
        if args.remote:
            # Local session inputs + durable SQLite; manually removed inputs
            # do not erase submission history, and external records may have
            # no journal. This does NOT enumerate the remote SCC station.
            identities = (
                [args.identity] if args.identity
                else sorted({p.name for p in packages} | {
                    str(row["measurement_id"]) for row in recorded
                })
            )
            if not identities:
                print("No locally known SCC measurement IDs for remote status.")
            else:
                client = _client(config)
                present = absent = errors = 0
                for identity in identities:
                    try:
                        record = client.measurement(identity)
                        reconcile_status(store, identity, record)
                        if args.backup:
                            if record is None:
                                raise ValueError("Cannot back up absent remote SCC measurement.")
                            from milgrau.scc.backup import (
                                archive_remote_measurement, verify_archive,
                            )

                            archive, digest = archive_remote_measurement(
                                client, record, state_root(config),
                            )
                            verify_archive(archive, digest)
                            print(
                                f"Remote SCC archive VERIFIED: {archive}; sha256={digest}; "
                                "no remote upload or deletion occurred."
                            )
                        if record is None:
                            absent += 1
                            print(f"remote {identity}: NOT FOUND")
                        else:
                            present += 1
                            system = record.get("system")
                            config_id = system.get("id") if isinstance(system, dict) else None
                            print(
                                f"remote {identity}: present; configuration={config_id}; "
                                f"start={record.get('start')}; stop={record.get('stop')}; "
                                f"queued={record.get('is_queued')}; "
                                f"running={record.get('is_being_processed')}; "
                                f"ELPP={record.get('elpp')}; ELDA={record.get('elda')}"
                            )
                        guidance = reconciliation_guidance(store, identity, record)
                        if guidance is not None:
                            print(f"ATTENTION {identity}: {guidance}")
                    except Exception as exc:
                        errors += 1
                        print(
                            f"remote {identity}: ERROR — {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        if args.identity:
                            raise
                if not args.identity:
                    print(
                        f"Remote SCC status summary: total={len(identities)}; "
                        f"present={present}; absent={absent}; errors={errors}; "
                        "no remote POST/DELETE."
                    )
                local_errors += errors
    return 1 if local_errors else 0


def _download_one(
    config: dict[str, Any], identity: str, client: Any, *,
    source_session: str | None,
) -> None:
    """Only store verified remote results; never delete SCC input artifacts."""
    from milgrau.scc.results import download_results

    destination, manifest = download_results(
        config, identity, client, source_session=source_session,
    )
    print(
        f"SCC downloads VERIFIED: {destination} | "
        f"products={', '.join(manifest['files'])} | "
        f"upload_owned={manifest['uploaded_by_milgrau']}; "
        "session SCC inputs retained."
    )


def _download(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.identity:
        _download_one(
            config, args.identity, _client(config),
            source_session=args.session,
        )
        return 0
    if args.session:
        raise ValueError("--session requires --id when downloading one measurement.")

    # A batch is precisely the union of locally staged packets and recorded
    # historical submissions, NOT every measurement hosted by the SCC station.
    # Local L0 identity is required for scientific provenance.
    from milgrau.scc.state import SubmissionStore, state_root
    from milgrau.level0.scc_staging import validate_pending_scc_package

    pending = {p.name: p for p in discover_pending(config)}
    with SubmissionStore(state_root(config)) as store:
        journal = {str(row["measurement_id"]): row for row in store.list()}
    identities = sorted(set(pending) | set(journal))
    if not identities:
        print("No locally known SCC measurements available for bulk download.")
        return 0
    client = _client(config)
    verified = not_ready = absent = failed = 0
    for index, identity in enumerate(identities, 1):
        prefix = f"[{index}/{len(identities)}] {identity}"
        try:
            # For old remote submissions without a local journal, infer the
            # scientific session ONLY from a hash-verified staged manifest.
            row = journal.get(identity)
            source_session = None
            if row is None:
                path = pending[identity]
                manifest = validate_pending_scc_package(path)
                if manifest["measurement_id"] != identity:
                    raise ValueError("SCC pending manifest measurement ID mismatch.")
                source_session = str(manifest["scientific_session_id"])
            remote = client.measurement(identity)
            if remote is None:
                absent += 1
                print(f"{prefix}: SKIPPED — not present on SCC.")
                continue
            if (
                remote.get("is_queued") or remote.get("is_being_processed")
                or remote.get("elpp") != 127 or remote.get("elda") != 127
            ):
                not_ready += 1
                print(
                    f"{prefix}: WAITING — SCC processing incomplete "
                    f"(ELPP={remote.get('elpp')}, ELDA={remote.get('elda')})."
                )
                continue
            # Batch downloads always preserve session-scoped input artifacts.
            _download_one(
                config, identity, client, source_session=source_session,
            )
            verified += 1
            print(f"{prefix}: VERIFIED — results and SCC inputs retained.")
        except Exception as exc:
            failed += 1
            print(
                f"{prefix}: ERROR — {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    print(
        f"SCC download summary: total={len(identities)}; verified={verified}; "
        f"waiting={not_ready}; absent={absent}; errors={failed}; "
        "all session SCC inputs retained."
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; no live network operations unless explicitly requested."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config()
        if not config.get("scc", {}).get("enabled") or "_scc_catalog" not in config:
            raise ValueError(
                "SCC integration is disabled. Set config.yaml scc.enabled: true."
            )
        if args.command == "upload":
            return _upload(config, args)
        if args.command == "status":
            return _status(config, args)
        if args.command == "download":
            return _download(config, args)
        parser.error(f"Unexpected SCC subcommand: {args.command}")
    except Exception as exc:
        print(f"SCC operation failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
