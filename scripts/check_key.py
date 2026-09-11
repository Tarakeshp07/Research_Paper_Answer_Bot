"""
Validate the Gemini API key before running anything expensive.

    python scripts/check_key.py

Makes one tiny call and translates the result into a plain diagnosis. Run this
whenever the pipeline starts failing on API calls — it tells you in two seconds
what would otherwise take five minutes of retry spam to work out.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config   # noqa: E402


DIAGNOSES = {
    "leaked": (
        "KEY REVOKED — Google flagged this key as PUBLICLY LEAKED.",
        [
            "This key is permanently dead. Retrying will never work.",
            "",
            "1. Revoke it now:  https://aistudio.google.com/apikey",
            "   Find the key and delete it.",
            "2. Create a new key there.",
            "3. Put the NEW key in .env (never in a notebook cell, never in code).",
            "4. Find where the old one leaked — run scripts/find_leaked_key.py",
            "   A key Google found publicly is usually in a git history or a",
            "   pushed .env. If you do not find and fix the source, the new key",
            "   will be flagged too.",
        ],
    ),
    "api_key_not_valid": (
        "KEY INVALID — Google does not recognise this key.",
        [
            "Usually a copy/paste problem:",
            "  - stray quotes or spaces in .env (use GOOGLE_API_KEY=AIza... with no quotes)",
            "  - the key was truncated when pasted",
            "  - you copied the project id instead of the key",
            "Create a fresh key at https://aistudio.google.com/apikey",
        ],
    ),
    "quota": (
        "QUOTA EXCEEDED — the key works but you have hit a rate or daily limit.",
        [
            "Free tier is roughly 15 requests/min and 1,500/day for Flash.",
            "Wait a minute and retry, or raise GEMINI_EMBED_SLEEP in src/config.py.",
        ],
    ),
    "permission": (
        "PERMISSION DENIED — the key exists but is not allowed to call this model.",
        [
            "Check that the Generative Language API is enabled for the key's project,",
            "and that no API restrictions are set on the key in Google Cloud console.",
        ],
    ),
}


def classify(err: str) -> str:
    e = err.lower()
    if "leaked" in e:
        return "leaked"
    if "api key not valid" in e or "api_key_invalid" in e:
        return "api_key_not_valid"
    if "quota" in e or "429" in e or "resource" in e and "exhaust" in e:
        return "quota"
    if "permission" in e or "403" in e:
        return "permission"
    return ""


def main() -> int:
    print("Checking Gemini API key...\n")

    try:
        key = config.get_google_api_key()
    except RuntimeError as exc:
        print("NO KEY FOUND\n")
        print(exc)
        return 1

    print(f"  key loaded: {len(key)} chars, ends ...{key[-4:]}")

    if not key.startswith("AIza"):
        print("  warning: Gemini keys normally start with 'AIza' — check what you pasted")

    print(f"  testing model: {config.LLM_MODEL}\n")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=0 — a bad key is not a transient failure, so retrying
        # only buries the real error under a wall of backoff messages.
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL, temperature=0, max_retries=0
        )
        resp = llm.invoke("Reply with the single word: ok")
        print("KEY WORKS")
        print(f"  model replied: {resp.content.strip()[:40]}")
        return 0

    except Exception as exc:
        msg = str(exc)
        kind = classify(msg)

        print("=" * 74)
        if kind:
            title, lines = DIAGNOSES[kind]
            print(title)
            print("=" * 74)
            for line in lines:
                print(line)
        else:
            print("API CALL FAILED")
            print("=" * 74)
            print(f"{type(exc).__name__}: {msg[:400]}")
        print("=" * 74)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
