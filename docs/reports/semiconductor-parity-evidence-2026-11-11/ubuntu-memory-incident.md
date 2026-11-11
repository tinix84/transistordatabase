# Ubuntu exhausted swap during the WSL launch failure

Codex · 2026-11-11 · Incident investigation, version 1.0

## Finding

Ubuntu recorded a global out-of-memory kill at 11:09:11 CET. Three Python processes held approximately 14.7 GiB of resident memory. The guest had exhausted all 8 GiB of swap. WSL also recorded two stalled connections. Guest memory pressure is a plausible contributor to the launch failure. The evidence does not establish why Windows returned E_UNEXPECTED or which Python commands ran.

## Scope and method

This record uses the persistent Ubuntu journal, the current process table, Linux memory counters and a source inspection of the XYZ energy import tests. All times are local on November 11. The incident boot ID is 8c337c0bc26547cd918976219d4b1e11. The recovery boot ID is 001247ca6cae43f1a73726fadc8baa64.

The incident journal contains multiple copies of kernel messages. Each copy has the same kernel source monotonic timestamp. The timeline uses the first journal receipt, with approximately one-second precision.

## Timeline

| Local time (CET) | Observation |
|---|---|
| 11:07:48 | The kernel reported an order-7 page allocation failure. Free swap was 8,045,356 kB. |
| 11:08:48 | The WSL relay reported an abnormally long virtual socket accept. |
| 11:09:11 | The kernel reported an order-4 allocation failure for Git. The OOM snapshot showed 8,388,608 kB total swap and 0 kB free swap. |
| 11:09:11 | A global OOM killed Python PID 35217, UID 1000, in /init.scope. It held 5,668,916 kB anonymous resident memory. |
| 11:09:48 | The same WSL relay reported a second abnormally long accept. |
| 12:19:43 | First journal entry in the recovery boot. |

## Process memory at the out-of-memory event

| PID | Kernel name | Resident memory (MiB) | Swap entries (MiB equivalent) |
|---:|---|---:|---:|
| 35118 | python | 8,426 | 1,632 |
| 35217 | python | 5,538 | 5,799 |
| 35166 | python | 1,109 | 243 |
| 35187 | python | 16 | 260 |
| 52610 | octave-gui | 45 | 39 |

The table uses 4 KiB per page and rounds to the nearest MiB. Resident totals can count shared pages more than once. The kernel reported 4,193,951 RAM pages, approximately 16 GiB. There were 23,230 free pages, approximately 91 MiB, in the snapshot adjacent to the kill.

## Attribution and recovery

No journal entries matched the killed PID as the emitting process. The OOM record contains the executable name and cgroup but omits the command arguments. No pytest or XYZ names appeared in the queried incident window. A review of the current XYZ tests found a single-cell input of 100,001 rows and a 321-row grid-boundary fixture. The implementation checks the grid product before matrix creation. The test source establishes neither historical execution nor memory use.

The kernel journal search for 11:06–11:12 returned no kernel panic, block I/O error, filesystem error or explicit disk-full message. The preceding evening's boot recorded storage command failures near its end. Their connection to the November 11 event remains unverified. The guest mounted its root filesystem read/write after recovery, with 857 GiB available inside ext4.

At about 12:25 CET, Ubuntu reported 14,697 MiB available of 15,990 MiB and all 8,192 MiB of swap free. A subsequent process snapshot found Python PID 10562 running uvicorn at 282.5 MiB RSS. No Node or Chromium process matched that snapshot. Process snapshots apply only to their collection time.

## Verification and next capture

1. Capture each memory-intensive command's peak RSS with /usr/bin/time -v during an isolated run.
2. Record process IDs, parent IDs and command names before running concurrent jobs.
3. Record guest memory and swap while the command runs.
4. If WSL launch fails again, capture the Windows WSL trace around one failed launch before recovery.

The Microsoft WSL trace can show the host service's error location. [Microsoft WSL debugging guide](https://github.com/microsoft/WSL/blob/master/doc/docs/debugging.md)

### Read-only commands used for verification

~~~powershell
wsl.exe -d Ubuntu -- journalctl --list-boots
wsl.exe -d Ubuntu -- journalctl -b 8c337c0bc26547cd918976219d4b1e11 -k --since '2026-11-11 11:07' --until '2026-11-11 11:11' --no-pager
wsl.exe -d Ubuntu -- journalctl -b 8c337c0bc26547cd918976219d4b1e11 _PID=35217 --no-pager
wsl.exe -d Ubuntu -- free --mebi
~~~