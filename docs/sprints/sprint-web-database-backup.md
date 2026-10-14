# Web database backup and restore sprint

## Goal

Provide a reachable local-web workflow for downloading a complete, versioned JSON backup and restoring it additively. A backup carries the raw legacy JSON records so legacy-only and unknown fields survive without passing through the partial core adapter.

## Boundaries

- Change only the existing TDB web application layer: FastAPI routes, the active Vue app, its API client, tests, and migration documentation.
- Do not change ntbees2, TDB core interfaces, adapters or converters, scientific models, persisted database records, or any real device data.
- Restore never overwrites or deletes an existing record. It validates the full input before writes, skips existing-name conflicts, and publishes each new file atomically under the cross-process web mutation lock using the no-replace helper.
- Keep the backup/restore controls in the mounted Vue application. The unmounted `DatabaseManager.vue` placeholder is not the implementation path.
- Harden the existing single-transistor JSON upload to create from the supplied raw object and reject an existing identity.
- Tests use only copied fixtures and temporary directories. No repository records are test targets.

## Backup format

```json
{
  "format": "transistordatabase.database-backup",
  "version": 1,
  "records": [
    { "name": "example", "...": "raw legacy JSON fields" }
  ]
}
```

Record objects remain raw JSON values. Export validates every stored JSON member and its filename identity; if any member is corrupt or mismatched, it returns a report instead of an incomplete backup. Restore rejects unsupported formats or versions, malformed records, unsafe or non-portable identities, case-insensitive duplicate names, numbers outside finite float range, non-UTF-8-encodable strings, and legacy records that fail the current reader before publishing any record.

## Acceptance criteria

- The active Vue navigation opens the backup/restore workflow and can download the server-generated full-database JSON backup.
- Selecting a file and validating it reports format/version, record counts, records eligible to add, and existing-name conflicts. Invalid files show actionable record-level errors.
- Restore requires an explicit browser confirmation. The result lists records added, name conflicts skipped, and any individual publication errors.
- Round-trip export/restore preserves legacy-only, unknown, and nested JSON values without core-model serialization.
- A corrupt database record prevents export from silently omitting it. A corrupt or duplicate input identity is rejected before writes; path traversal, slash/backslash, Windows drive-relative, reserved device names, invalid Windows filename characters, trailing dots/spaces, and NTFS alternate-data-stream identities are rejected. Generated paths are checked for containment under the configured repository using Windows path semantics and the current filesystem's resolved paths.
- Preview and restore apply the same case-insensitive existing-name conflict rules. Numbers that overflow or underflow finite float representation and strings that cannot be UTF-8 encoded are rejected during full prevalidation before any record is written.
- Existing-name conflicts remain byte-for-byte unchanged. New files are written atomically without replacement while holding the existing cross-process mutation lock.
- Single-record JSON upload has the same raw-preserving, create-only behavior.
- Focused backend and frontend tests and the web production build pass. The broader suite's unrelated existing failures are recorded separately.
- No Qt or ntbees2 code/data changes are introduced.

## Validation evidence

Run from `D:\repos\transistordatabase-sprint2` unless noted:

- `D:\codex-envs\tdb-web-lifecycle\Scripts\python.exe -m pytest --confcutdir=tests/web_database_backup tests/web_database_backup/test_api.py tests/test_web_database_lifecycle_api.py -q` — **65 passed**, with one Starlette `httpx` deprecation warning. Coverage includes raw round-trip, corrupt export reporting, conflict bytes unchanged, full-file prevalidation, duplicate-key and case-insensitive duplicate rejection, unsupported format/version, traversal and Windows filename safety, finite-number overflow/underflow and unpaired-surrogate rejection before writes, repository path containment under local and Windows path semantics, create-only upload, and preview/restore conflict consistency. All record files are fixture copies under pytest temporary directories; Windows path regression checks use `PureWindowsPath` without creating outside files.
- From `transistordatabase\gui_web`, `npm run test:run -- src/tests/unit/DatabaseBackupManager.test.js src/tests/unit/App.test.js -t "DatabaseBackupManager|database backup"` — **8 passed, 30 skipped**. The focused tests cover App navigation, backup download, corrupt-record download errors, file validation, restore confirmation/cancellation, restore results, preserving the same selected file while validation or restore is in flight, and clearing stale validation after selecting a different file.
- From `transistordatabase\gui_web`, `npm run build` — **passed**. Vite reports its existing warning that the minified JavaScript chunk is larger than 500 kB.
- No real database records, Qt components, ntbees2 files, TDB core interfaces/adapters/converters, or scientific models were changed or used as test targets. No Playwright output directories were used or modified.

Restore validates the entire backup before starting writes, but publication is atomic per record rather than transactional across the whole backup. If a filesystem error occurs after earlier records were published, the API reports the added and failed names; a retry skips the records already present. The existing no-replace helper supports Windows and Linux and fails safely on other platforms.

## Remaining migration gaps

This sprint covers full-database raw JSON backup/restore and the existing single-record JSON upload. It does not establish parity for PLECS/catalog imports, device-specific simulation exports, settings migration, saved-curve editing, transient thermal curves, DPT measurement handling, or other desktop-only capabilities. Those remain tracked by the [topology migration plan](sprint-web-topology-migration.md); this sprint does not close full local-web replacement.
