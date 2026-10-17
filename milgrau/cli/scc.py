"""Four-command operator CLI for optional SCC interoperability.

12.4 enables local dry-run, read-only status and one explicitly confirmed
upload. Download/cleanup require the durable submission registry in 12.5/12.6.
"""

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
        help="Destructively replace existing SCC record under same ID after VERIFIED remote backup.",
    )
    action = upload.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run", action="store_true",
        help="Validate pending packages locally; never connect to the SCC.",
    )
    action.add_argument(
        "--confirm", action="store_true",
        help="Authorize ONE real upload. Requires --id and SCC environment credentials.",
    )

    status = sub.add_parser("status", help="List local pending packets or query remote SCC.")
    status.add_argument("--id", dest="identity", help="Filter by Measurement_ID.")
    status.add_argument(
        "--remote", action="store_true", help="Also query remote SCC (requires --id and credentials).",
    )
    status.add_argument(
        "--backup", action="store_true",
        help="Read-only download and verification of existing SCC remote inputs/results; requires --remote --id.",
    )
    sub.add_parser("download", help="Reserved for Phase 12.6; never silently downloads.")
    sub.add_parser("cleanup", help="Reserved for Phase 12.6; never silently deletes.")
    return parser


def _client(config: dict[str, Any]):
    """Lazy-load HTTP extras only for explicit remote operations."""
    from milgrau.scc.client import SCCClient, SCCCredentials

    catalog = config.get("_scc_catalog", {})
    server = catalog.get("server", {})
    if not isinstance(server, dict) or not server.get("base_url"):
        raise ValueError("Set scc.yaml server.base_url before remote SCC operations.")
    upload = config["scc"].get("upload", {})
    return SCCClient(
        server["base_url"],
        SCCCredentials.from_environment(),
        timeout=float(upload.get("timeout_seconds", 60)),
        retries=upload.get("retries", 2),
    )


def _upload(config: dict[str, Any], args: argparse.Namespace) -> int:
    packages = discover_pending(config, args.identity)
    if not packages:
        print("No SCC packages are staged in .cache/scc/pending/.")
        return 0 if args.dry_run else 1
    if args.replace_existing and (args.dry_run or not args.confirm or not args.identity):
        raise ValueError(
            "--replace-existing requires --id ID --confirm; it cannot run in dry-run."
        )
    if args.confirm and (not args.identity or len(packages) != 1):
        raise ValueError(
            "Actual upload requires --id MEASUREMENT_ID --confirm "
            "(one measurement at a time until persistent state is implemented)."
        )
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

    # Validate full package before touching any network resources.
    path = Path(packages[0])
    validate_upload_package(path)
    from milgrau.scc.workflow import submit

    client = _client(config)
    result = submit(
        config, path, client, replace_existing=args.replace_existing,
    )
    print(
        f"SCC API confirmed {result['measurement_id']}: {result['remote_page']}. "
        "Submission is journalled in .state/scc; cached files remain "
        "until safe cleanup is available in Phase 12.6."
    )
    return 0


def _status(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.remote and not args.identity:
        raise ValueError("Remote status requires --id MEASUREMENT_ID.")
    if args.backup and not args.remote:
        raise ValueError("Remote archive --backup requires --id ID --remote.")
    # Remote measurements may outlive the disposable local staging cache.
    packages = discover_pending(config)
    if args.identity is not None:
        packages = [path for path in packages if path.name == args.identity]
    for path in packages:
        try:
            manifest = validate_upload_package(path)
            print(
                f"pending {path.name} ({manifest['solar_regime']} / "
                f"{manifest['scc_configuration_id']}); local integrity OK"
            )
        except Exception as exc:
            print(f"invalid pending package {path.name}: {exc}", file=sys.stderr)
            return 1
    if not packages:
        print("No matching local SCC pending packets.")
    from milgrau.scc.state import SubmissionStore, state_root
    from milgrau.scc.workflow import reconcile_status

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
            client = _client(config)
            record = client.measurement(args.identity)
            reconcile_status(store, args.identity, record)
            if args.backup:
                if record is None:
                    raise ValueError(
                        "Cannot back up absent remote SCC measurement."
                    )
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
                print(f"remote {args.identity}: NOT FOUND")
            else:
                system = record.get("system")
                config_id = system.get("id") if isinstance(system, dict) else None
                print(
                    f"remote {args.identity}: present; configuration={config_id}; "
                    f"start={record.get('start')}; stop={record.get('stop')}; "
                    f"queued={record.get('is_queued')}; "
                    f"running={record.get('is_being_processed')}; "
                    f"ELPP={record.get('elpp')}; ELDA={record.get('elda')}"
                )
                row = store.get(args.identity)
                if row is not None and row['state'] == 'uncertain':
                    print(
                        "ATTENTION: journal remains uncertain; no automatic "
                        "retry or replacement is permitted."
                    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; no live network operations unless explicitly requested."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command in {"download", "cleanup"}:
        print(
            f"milgrau-scc {args.command} is not enabled yet. "
            "Requires persistent, verified remote submission state "
            "(Phases 12.5–12.6); no files were modified.",
            file=sys.stderr,
        )
        return 2
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
        parser.error(f"Unexpected SCC subcommand: {args.command}")
    except Exception as exc:
        print(f"SCC operation failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
