"""Four-command SCC CLI: upload, status, verified download and safe cleanup."""

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
        help="Replace existing SCC IDs only after verified backup; with no --id applies sequentially to the batch.",
    )
    upload.add_argument(
        "--resume-after-delete", action="store_true",
        help="Exact-ID recovery ONLY after uncertain DELETE, verified old archive and SCC 404; sends one upload, no deletion.",
    )
    upload.add_argument(
        "--retry-after-upload", action="store_true",
        help="One explicit same-ID upload-only retry after recovery_upload_uncertain; "
             "requires --id ID --confirm, never deletes or runs in batch.",
    )
    upload.add_argument(
        "--retry-after-lr-fix", action="store_true",
        help="One explicitly confirmed same-ID POST after journalled LR filename repair; "
             "never deletes, never retries automatically or runs in batch.",
    )
    action = upload.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run", action="store_true",
        help="Validate pending packages locally; never connect to the SCC.",
    )
    action.add_argument(
        "--migrate-cache", action="store_true",
        help="Migrate staged raw.nc to Measurement_ID.nc for one or all IDs; offline only.",
    )
    action.add_argument(
        "--repair-lr-cache", action="store_true",
        help="Offline: fix rejected LR filename and raw LR_File_Name; retain original "
             "cache bytes and audit journal hashes, no SCC POST.",
    )
    action.add_argument(
        "--confirm", action="store_true",
        help="Authorize SCC upload: one --id, or all local pending IDs sequentially if --id omitted.",
    )

    sounding = sub.add_parser(
        "sounding",
        help="Preview or explicitly add LIPANCORA-observed soundings to legacy pending packets (offline).",
    )
    sounding.add_argument("--id", dest="identity", help="Optional single SCC Measurement_ID.")
    sounding_action = sounding.add_mutually_exclusive_group(required=True)
    sounding_action.add_argument(
        "--dry-run", action="store_true",
        help="Inspect legacy packages/L1 cache, no changes to pending files or SCC.",
    )
    sounding_action.add_argument(
        "--confirm", action="store_true",
        help="Update eligible unsubmitted cache packets from verified L1 observations (no SCC requests).",
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
    download = sub.add_parser(
        "download", help="Read-only SCC result download into scientific session/scc-results."
    )
    download.add_argument("--id", dest="identity", required=True, help="Exact SCC Measurement_ID.")
    download.add_argument(
        "--session", help="Scientific Session_ID for a remote record not uploaded by MILGRAU."
    )
    download.add_argument(
        "--keep-cache", action="store_true",
        help="Keep eligible staging files even if cleanup.policy is after_download."
    )
    cleanup = sub.add_parser(
        "cleanup", help="Remove only verified pending SCC cache for a journal-confirmed upload."
    )
    cleanup.add_argument("--id", dest="identity", required=True)
    approval = cleanup.add_mutually_exclusive_group(required=True)
    approval.add_argument("--dry-run", action="store_true")
    approval.add_argument("--confirm", action="store_true")
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
        return 0 if args.dry_run or args.migrate_cache or args.repair_lr_cache else 1
    if args.resume_after_delete and (
        not args.confirm or not args.identity
        or args.replace_existing or args.retry_after_upload
        or args.retry_after_lr_fix
    ):
        raise ValueError(
            "--resume-after-delete requires exactly --id ID --confirm; "
            "never combine it with batch mode or --replace-existing."
        )
    if args.retry_after_upload and (
        not args.confirm or not args.identity
        or args.replace_existing or args.resume_after_delete
        or args.retry_after_lr_fix
    ):
        raise ValueError(
            "--retry-after-upload requires exactly --id ID --confirm; "
            "never use it with batch, --replace-existing or --resume-after-delete."
        )
    if args.retry_after_lr_fix and (
        not args.confirm or not args.identity
        or args.replace_existing or args.resume_after_delete
        or args.retry_after_upload
    ):
        raise ValueError(
            "--retry-after-lr-fix requires --id ID --confirm, "
            "never batch/replace/another retry mode."
        )
    if args.migrate_cache:
        if (args.replace_existing or args.resume_after_delete
                or args.retry_after_upload or args.retry_after_lr_fix):
            raise ValueError("Cache migration never combines with remote upload options.")
        from milgrau.scc.migrate import migrate_pending

        changed = 0
        for path in packages:
            destination, migrated = migrate_pending(config, path.name)
            print(f"{'MIGRATED' if migrated else 'ALREADY CURRENT'} {destination.name}")
            changed += int(migrated)
        print(f"Local cache: {changed} migrated, {len(packages)-changed} already current; "
              "no SCC network requests, raw bytes or LR modified.")
        return 0
    if args.repair_lr_cache:
        if args.replace_existing or args.resume_after_delete or args.retry_after_upload:
            raise ValueError("Local LR cache repair cannot authorize any SCC operation.")
        from milgrau.scc.lr_repair import repair_lr_cache

        changed = 0
        for path in packages:
            _, repaired = repair_lr_cache(config, path.name)
            print(f"{'REPAIRED' if repaired else 'ALREADY CURRENT'} LR {path.name}")
            changed += int(repaired)
        print(
            f"Local LR repair: {changed} repaired, "
            f"{len(packages) - changed} already current; "
            "SCC not contacted, original cache saved."
        )
        return 0
    if args.replace_existing and args.dry_run:
        raise ValueError(
            "--replace-existing must be used with --confirm; "
            "offline --dry-run does not inspect remote SCC IDs."
        )
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
            f"failed={report.failed}; unprocessed={report.remaining}."
        )
        if report.stopped_at:
            print(
                f"Batch stopped at {report.stopped_at}. "
                "Review the SCC status and durable journal; do not blindly restart."
            )
        print("No staged cache, scientific data or durable SCC archive was removed.")
        return 0 if report.successful else 1

    # Single-ID operation uses the same guarded workflow as before.
    path = Path(packages[0])
    validate_upload_package(path)
    from milgrau.scc.workflow import (
        resume_after_uncertain_delete, retry_rejected_recovery_upload, submit,
    )

    client = _client(config)
    if args.resume_after_delete:
        result = resume_after_uncertain_delete(config, path, client)
    elif args.retry_after_upload:
        result = retry_rejected_recovery_upload(config, path, client)
    elif args.retry_after_lr_fix:
        result = retry_rejected_recovery_upload(
            config, path, client, after_lr_fix=True,
        )
    else:
        result = submit(
            config, path, client, replace_existing=args.replace_existing,
        )
    print(
        f"SCC API confirmed {result['measurement_id']}: {result['remote_page']}. "
        "Submission is journalled in .state/scc; cached files remain "
        "unchanged until an explicitly configured cleanup after download."
    )
    return 0


def _sounding(config: dict[str, Any], args: argparse.Namespace) -> int:
    from milgrau.scc.sounding_backfill import backfill_pending_soundings

    report, code = backfill_pending_soundings(
        config, identity=args.identity, confirm=args.confirm,
    )
    counts: dict[str, int] = {}
    for item in report:
        print(f"{item.status} {item.identity}: {item.reason}")
        counts[item.status] = counts.get(item.status, 0) + 1
    summary = ", ".join(f"{key.lower()}={val}" for key, val in sorted(counts.items()))
    print(
        f"SCC sounding {'backfill' if args.confirm else 'preview'}: "
        f"total={len(report)}; {summary or 'none'}. "
        "No SCC network calls, no submission journal changes."
    )
    return code


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


def _download(config: dict[str, Any], args: argparse.Namespace) -> int:
    from milgrau.scc.results import download_results
    from milgrau.scc.cleanup import cleanup_pending

    destination, manifest = download_results(
        config, args.identity, _client(config), source_session=args.session,
    )
    print(
        f"SCC downloads VERIFIED: {destination} | "
        f"products={', '.join(manifest['files'])} | "
        f"upload_owned={manifest['uploaded_by_milgrau']}"
    )
    if not args.keep_cache and config["scc"].get("cleanup", {}).get("policy") == "after_download":
        if manifest["uploaded_by_milgrau"]:
            result = cleanup_pending(config, args.identity, dry_run=False)
            print(
                f"SCC pending cache cleanup: "
                f"{'already done' if result['already_cleaned'] else 'verified and removed'} "
                f"for {args.identity}; persistent results and journal preserved."
            )
        else:
            print(
                "SCC pending cache retained: remote record was not confirmed as "
                "a MILGRAU upload."
            )
    return 0


def _cleanup(config: dict[str, Any], args: argparse.Namespace) -> int:
    from milgrau.scc.cleanup import cleanup_pending

    result = cleanup_pending(config, args.identity, dry_run=not args.confirm)
    if result["already_cleaned"]:
        print(f"SCC pending cache already cleaned: {args.identity}")
    elif args.confirm:
        print(f"SCC pending cache removed: {args.identity}; results/journal preserved.")
    else:
        print(
            f"DRY RUN: eligible pending SCC package {result['staged']} "
            "would be removed; nothing was deleted."
        )
    return 0


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
        if args.command == "sounding":
            return _sounding(config, args)
        if args.command == "status":
            return _status(config, args)
        if args.command == "download":
            return _download(config, args)
        if args.command == "cleanup":
            return _cleanup(config, args)
        parser.error(f"Unexpected SCC subcommand: {args.command}")
    except Exception as exc:
        print(f"SCC operation failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
