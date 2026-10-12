# Independent review of the new code (before the release PR)

Six independent reviewers read the code added in this PR — one each for HTTP/API security, self-update and CI supply chain,
VirtualGPU decision logic, VirtualGPU runtime safety, the web UI, and the engines/learning stack. Every finding then went to a
skeptic whose job was to refute it (two skeptics for anything rated high); a finding survived only if a skeptic reproduced it or
pointed at the exact lines that cause it. 29 agents ran; **20 findings were confirmed and 1 was refuted.** All 20 are fixed in this PR,
each with a regression test, and the fixes were re-checked on Python 3.10 and 3.13 and (for the UI) in a real browser.

| # | Area | Finding (as confirmed) | Fix |
|---|---|---|---|
| 1 | tools | With tools + web research on and default confirmations, a web page could make the model read `~/.duckstoken/settings.json` (API keys) or `~/.ssh` and send it out in a URL, with no dialog | App data folder never readable/writable; credential locations always need approval; web tools ask first by default |
| 2 | tools | The write approval showed only a path and a character count; the settings file was writable by the AI | Dialog shows the exact content; data folder denied; "always allow" never offered for writes/commands/secrets |
| 3 | settings | Export wrote API keys in plaintext; Import silently applied PC-access and launcher settings | Export blanks keys; import / profile load need confirmation for risky changes (server-enforced) |
| 4 | app | "Quit" and "Install & restart" left the process running (the update helper then gave up) | `shutdown()` ends the native shell and runs once |
| 5 | packaging | PyInstaller flags bundled only pywebview's *data*, so no released exe could ever open its native window | Shared `tools/build_exe.py` with `--collect-all webview`; `Duckstoken.exe --selftest` verified in CI against the portable exe and the installed app |
| 6 | updater | Auto-install launched the installer two or more times | Applying is one-shot; installer `SetupMutex` |
| 7 | recovery | Out-of-memory crashes at load time were classified "bad model", so recovery gave up and blamed the file | Out-of-memory is classified first; generic "failed to load model" lines no longer decide anything |
| 8 | optimizer | Re-running the optimizer after Apply erased the saved previous values, so Undo failed | The earlier record's undo data is carried over |
| 9 | tuner | Auto-tune placed ~2× the available VRAM of gpt-oss-120b on the GPU | Placement comes from the memory planner; the load gate also fits GPU offload to VRAM for that start only |
| 10 | planner | What-if (HDD/NVMe) disks were matched against the real model's Windows drive letter and silently ignored | What-if hardware uses its own drive |
| 11 | staging | A full or read-only cache drive made `StoreError` escape and blocked starting the model | Handled everywhere; the cache can never block a start |
| 12 | runtime | A second Start during "prewarm before start" orphaned the first cancel event, so Stop did not stop the server | Second Start refused (409); Stop cancels the run in flight |
| 13 | ui | `append()` printed `[object HTMLDivElement]` / `null` (blocker lists, dashboard, engine model list) | One flatten-and-skip `append`, verified in the browser |
| 14 | ui | A finished job was re-rendered and re-fetched every 1.5 s forever; copy jobs showed as a failed optimizer run | Acts once per job id and kind |
| 15 | ui | A failed benchmark showed `0.0 / 0.00 / 0.00` for the first run | Shows "n/a" |
| 16 | ui | The danger-switch confirmation fired in the safe direction and not the unsafe one | Fires when moving away from the default; cancel restores the switch |
| 17 | ui | Switching chats during a tool round sent the *other* chat's history to the model | The request uses the chat being answered; DOM writes are guarded (browser test with mocked tool rounds) |
| 18 | db | Every HTTP request thread leaked a SQLite connection (≈300 KB and two handles each) | Connections of finished threads are closed |
| 19 | web | The fetch deadline was only checked between reads; a server trickling bytes held a chat turn indefinitely | Deadline enforced on the socket (also the header phase) |
| 20 | feedback | Deleting a chat left its transcripts, queued jobs and training examples; no retention | Deleting a chat purges them; optional age-based retention for unrated logs |

**Refuted:** "model servers listen on fixed loopback ports with no API key by default" — a documented design choice; the per-launch
token protects the Duckstoken API, and the model server's key can be set in Settings.

## What this review did not cover

* Real Windows hardware: ctypes struct layouts, the native window/tray, the installer on a physical PC.
* A real GPU or a real gpt-oss-120b load.
* Prompt injection is mitigated (approvals, protected locations, ask-first web tools), not solved: with Full PC access, "always allow" on
  reads and confirmations off, a model that follows hostile text can still do harm. The README says so.
* Fuzzing of the GGUF and tokenizer parsers beyond what the reviewers tried by hand.
