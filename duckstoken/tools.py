"""The 'use my PC' tools the model may call, with path policy, approvals and an audit log."""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable

from . import hardware, paths

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

TOOLS: dict[str, dict] = {
    "list_dir": {
        "setting": "tool_list_dir", "confirm": "confirm_reads",
        "description": "List the files and folders inside a directory on the user's PC.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path, e.g. C:\\Users\\me\\Documents"}},
            "required": ["path"]}},
    "read_file": {
        "setting": "tool_read_file", "confirm": "confirm_reads",
        "description": "Read a text file from the user's PC.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "Byte offset to start at (default 0)"},
            "max_bytes": {"type": "integer", "description": "How many bytes to read at most"}},
            "required": ["path"]}},
    "find_files": {
        "setting": "tool_find_files", "confirm": "confirm_reads",
        "description": "Search for files by name pattern (e.g. *.pdf) under a folder.",
        "parameters": {"type": "object", "properties": {
            "root": {"type": "string"}, "pattern": {"type": "string", "description": "Glob such as *.txt"},
            "max_results": {"type": "integer"}}, "required": ["root", "pattern"]}},
    "system_info": {
        "setting": "tool_system_info", "confirm": None,
        "description": "Get CPU, RAM, GPU and disk information about this PC.",
        "parameters": {"type": "object", "properties": {}}},
    "write_file": {
        "setting": "tool_write_file", "confirm": "confirm_writes",
        "description": "Create or overwrite a text file (or append to it) on the user's PC.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "append": {"type": "boolean", "description": "Append instead of overwrite"}},
            "required": ["path", "content"]}},
    "run_command": {
        "setting": "tool_run_command", "confirm": "confirm_commands",
        "description": ("Run a command in " + ("PowerShell" if os.name == "nt" else "the shell") +
                        " on the user's PC and return its output."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "cwd": {"type": "string", "description": "Working directory"}},
            "required": ["command"]}},
    "fetch_url": {
        "setting": "tool_fetch_url", "confirm": "confirm_fetch",
        "description": "Download a web page or text resource over http(s) and return it as text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
}


class ToolError(Exception):
    pass


def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.expandvars(os.path.expanduser(p))))


def _inside(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # different drives
        return False


class ToolRunner:
    def __init__(self, get_settings: Callable[[], dict]):
        self.get = get_settings

    # ---------------------------------------------------------------- schema
    def definitions(self) -> list[dict]:
        s = self.get()
        if not s["tools_enabled"]:
            return []
        return [{"type": "function", "function": {"name": n, "description": t["description"],
                                                  "parameters": t["parameters"]}}
                for n, t in TOOLS.items() if s[t["setting"]]]

    # ---------------------------------------------------------------- policy
    def check_path(self, raw: str, write: bool = False) -> str:
        if not raw or not isinstance(raw, str):
            raise ToolError("A path is required")
        s = self.get()
        p = _norm(raw)
        if not s["full_pc_access"]:
            roots = s["allowed_roots"] or [str(Path.home())]
            if not any(_inside(p, _norm(r)) for r in roots):
                raise ToolError(f"Access denied: '{raw}' is outside the allowed folders. Enable 'Full PC access' "
                                "or add the folder under PC access.")
        if write and s["protect_system_paths"]:
            for b in s["blocked_paths"]:
                if _inside(p, _norm(b)):
                    raise ToolError(f"Writing is blocked inside protected folder {b}")
        return p

    def describe(self, name: str, args: dict) -> str:
        if name == "run_command":
            return f"Run command:\n{args.get('command', '')}" + (f"\n(in {args['cwd']})" if args.get("cwd") else "")
        if name == "write_file":
            mode = "Append to" if args.get("append") else "Write"
            return f"{mode} file: {args.get('path')}\n({len(str(args.get('content', '')))} characters)"
        if name == "fetch_url":
            return f"Fetch: {args.get('url')}"
        key = args.get("path") or args.get("root") or ""
        return f"{name.replace('_', ' ').capitalize()}: {key}"

    def validate_call(self, name: str) -> dict:
        s = self.get()
        spec = TOOLS.get(name)
        if not spec or not s["tools_enabled"] or not s[spec["setting"]]:
            raise ToolError(f"Tool '{name}' is not enabled")
        return spec

    def needs_approval(self, name: str) -> bool:
        spec = self.validate_call(name)
        return bool(spec["confirm"] and self.get()[spec["confirm"]])

    # ------------------------------------------------------------- execution
    def execute(self, name: str, args: dict) -> dict:
        s = self.get()
        t0 = time.time()
        try:
            self.validate_call(name)
            fn = getattr(self, f"_t_{name}")
            out = fn(args or {})
            text = out if isinstance(out, str) else json.dumps(out, indent=1)
            cap = s["max_output_chars"]
            if len(text) > cap:
                text = text[:cap] + f"\n...[truncated {len(text) - cap} characters]"
            res = {"ok": True, "output": text}
        except ToolError as e:
            res = {"ok": False, "output": f"Error: {e}"}
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "output": f"Error: {e.__class__.__name__}: {e}"}
        if s["audit_log"]:
            self._audit(name, args, res, time.time() - t0)
        return res

    def _audit(self, name, args, res, secs) -> None:
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": name, "ok": res["ok"],
               "seconds": round(secs, 2), "args": {k: (str(v)[:300]) for k, v in (args or {}).items()}}
        try:
            with open(paths.audit_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def audit_tail(self, n: int = 200) -> list[dict]:
        try:
            lines = paths.audit_file().read_text(encoding="utf-8").splitlines()[-n:]
        except OSError:
            return []
        out = []
        for ln in reversed(lines):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out

    # ------------------------------------------------------------------ tools
    def _t_list_dir(self, a: dict):
        p = self.check_path(a.get("path", ""))
        if not os.path.isdir(p):
            raise ToolError("Not a directory")
        rows = []
        with os.scandir(p) as it:
            for e in sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower())):
                try:
                    st = e.stat(follow_symlinks=False)
                    rows.append(f"{'[dir] ' if e.is_dir(follow_symlinks=False) else '      '}{e.name}"
                                f"{'' if e.is_dir(follow_symlinks=False) else f'  ({st.st_size} bytes)'}")
                except OSError:
                    rows.append(f"      {e.name}  (unreadable)")
                if len(rows) >= 1000:
                    rows.append("...[more entries not shown]")
                    break
        return "\n".join(rows) or "(empty)"

    def _t_read_file(self, a: dict):
        p = self.check_path(a.get("path", ""))
        if not os.path.isfile(p):
            raise ToolError("Not a file")
        s = self.get()
        limit = min(int(a.get("max_bytes") or s["max_read_bytes"]), s["max_read_bytes"])
        with open(p, "rb") as f:
            f.seek(max(int(a.get("offset") or 0), 0))
            data = f.read(limit)
        if b"\0" in data[:4096]:
            raise ToolError("This looks like a binary file; only text files can be read.")
        return data.decode("utf-8", "replace")

    def _t_find_files(self, a: dict):
        root = self.check_path(a.get("root", ""))
        pat, limit = a.get("pattern") or "*", min(int(a.get("max_results") or 200), 1000)
        deadline, hits = time.time() + 15, []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith("$")]
            for f in files:
                if fnmatch.fnmatch(f.lower(), pat.lower()):
                    hits.append(os.path.join(d, f))
                    if len(hits) >= limit:
                        return "\n".join(hits) + "\n...[limit reached]"
            if time.time() > deadline:
                return "\n".join(hits) + "\n...[search timed out]"
        return "\n".join(hits) or "No matches"

    def _t_system_info(self, a: dict):
        hw = hardware.get_hardware()
        hw["platform"] = platform.platform()
        return hw

    def _t_write_file(self, a: dict):
        p = self.check_path(a.get("path", ""), write=True)
        content = str(a.get("content", ""))
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a" if a.get("append") else "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return f"{'Appended' if a.get('append') else 'Wrote'} {len(content)} characters to {p}"

    def _t_run_command(self, a: dict):
        cmd = str(a.get("command", "")).strip()
        if not cmd:
            raise ToolError("A command is required")
        s = self.get()
        cwd = self.check_path(a["cwd"]) if a.get("cwd") else str(Path.home())
        if os.name == "nt":
            argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
        else:
            argv = ["/bin/sh", "-c", cmd]
        try:
            r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, errors="replace",
                               timeout=s["command_timeout"], creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {s['command_timeout']} s")
        return f"exit code {r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"

    def _t_fetch_url(self, a: dict):
        url = str(a.get("url", ""))
        if not url.lower().startswith(("http://", "https://")):
            raise ToolError("Only http(s) URLs are allowed")
        req = urllib.request.Request(url, headers={"User-Agent": "Duckstoken/0.1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read(self.get()["max_read_bytes"])
        return data.decode("utf-8", "replace")
