"""
Find where a Google API key leaked.

    python scripts/find_leaked_key.py                 # scan D:\\capstone
    python scripts/find_leaked_key.py C:\\some\\path    # scan somewhere else

Scans working files AND git history for anything shaped like a Gemini key
(AIza followed by 35 characters). Google's secret scanner found one of yours in
a public place; unless you find and remove the source, a replacement key will be
flagged the same way.

Keys found are shown MASKED — only the first 8 and last 4 characters — so the
output of this script is safe to share.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

KEY_RE = re.compile(rb"AIza[0-9A-Za-z_\-]{35}")

SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
             ".pytest_cache", "data", ".ipynb_checkpoints"}

TEXT_SUFFIXES = {".py", ".ipynb", ".md", ".txt", ".json", ".yaml", ".yml",
                 ".env", ".cfg", ".ini", ".toml", ".js", ".ts", ".tsx",
                 ".html", ".sh", ".ps1", ".bat", ""}


def mask(key: str) -> str:
    return f"{key[:8]}...{key[-4:]}"


def scan_files(root: Path) -> list[tuple[Path, str]]:
    hits: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 20_000_000:
                continue
            blob = path.read_bytes()
        except OSError:
            continue
        for m in KEY_RE.findall(blob):
            hits.append((path, m.decode()))
    return hits


def git_repos(root: Path) -> list[Path]:
    return [p.parent for p in root.rglob(".git") if p.is_dir()]


def scan_git_history(repo: Path) -> list[tuple[str, str]]:
    """
    Search every blob in the repo's history. A key deleted from the working tree
    still sits in history forever, and that history is what got pushed.
    """
    hits: list[tuple[str, str]] = []
    try:
        rev_list = subprocess.run(
            ["git", "rev-list", "--all", "--objects"],
            cwd=repo, capture_output=True, timeout=120,
        )
        if rev_list.returncode != 0:
            return hits

        grep = subprocess.run(
            ["git", "grep", "-I", "-E", "-h", "AIza[0-9A-Za-z_-]{35}",
             "--", "."],
            cwd=repo, capture_output=True, timeout=120,
        )
        for m in KEY_RE.findall(grep.stdout or b""):
            hits.append(("working tree", m.decode()))

        all_grep = subprocess.run(
            ["git", "log", "--all", "-p", "-S", "AIza", "--pretty=format:COMMIT %H %s"],
            cwd=repo, capture_output=True, timeout=300,
        )
        out = all_grep.stdout or b""
        current = "unknown commit"
        for line in out.split(b"\n"):
            if line.startswith(b"COMMIT "):
                current = line.decode(errors="replace")[7:100]
            for m in KEY_RE.findall(line):
                hits.append((current, m.decode()))

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return hits


def has_remote(repo: Path) -> str:
    try:
        r = subprocess.run(["git", "remote", "-v"], cwd=repo,
                           capture_output=True, timeout=30)
        out = (r.stdout or b"").decode(errors="replace").strip()
        return out.split("\n")[0] if out else ""
    except Exception:
        return ""


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\capstone")
    root = root.resolve()

    if not root.exists():
        print(f"Path not found: {root}")
        return 1

    print(f"Scanning: {root}\n")
    print("=" * 74)
    print("1. WORKING FILES")
    print("=" * 74)

    file_hits = scan_files(root)
    if file_hits:
        for path, key in file_hits:
            rel = path.relative_to(root) if root in path.parents else path
            print(f"  FOUND  {mask(key)}  in  {rel}")
    else:
        print("  no keys found in working files")

    print()
    print("=" * 74)
    print("2. GIT HISTORY  (this is usually where the leak is)")
    print("=" * 74)

    repos = git_repos(root)
    if not repos:
        print("  no git repositories found")

    for repo in repos:
        remote = has_remote(repo)
        rel = repo.relative_to(root) if repo != root else "."
        print(f"\n  repo: {rel}")
        print(f"  remote: {remote or '(none — never pushed)'}")

        hits = scan_git_history(repo)
        if hits:
            seen = set()
            for where, key in hits:
                if (where, key) in seen:
                    continue
                seen.add((where, key))
                print(f"    FOUND  {mask(key)}  in  {where}")
            if remote:
                print("    ^^ THIS REPO HAS A REMOTE. If it is public, this is your leak.")
        else:
            print("    no keys found in history")

    print()
    print("=" * 74)
    print("WHAT TO DO")
    print("=" * 74)
    print("""
If a key was found in a repo that has a remote:

  1. The key is already public. Revoke it at https://aistudio.google.com/apikey
     Deleting the file does NOT help — the key stays in git history.

  2. If the repo is public and you do not need its history, the simplest fix is
     to make the repository private (or delete it) on GitHub.

  3. To keep the repo public, you must rewrite history to purge the key:
         pip install git-filter-repo
         git filter-repo --path .env --invert-paths --force
     then force-push. Anyone who cloned it still has the old key, so revoking
     in step 1 is what actually protects you — history rewriting is cleanup.

  4. Make sure .env is in .gitignore in EVERY project before the next commit:
         git check-ignore -v .env
     Should print a .gitignore line. If it prints nothing, .env is NOT ignored.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
