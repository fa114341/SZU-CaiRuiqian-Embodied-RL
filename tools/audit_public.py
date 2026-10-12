"""Fail closed on common public-release hazards in this curated repository."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BAD_NAMES = {"T800_CONTEXT.md", "id_rsa", "id_ed25519"}
BAD_EXTENSIONS = {".ckpt", ".pth", ".safetensors", ".pem", ".key"}
BAD_TEXT = {
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "private endpoint": re.compile(r"(?i)azy6\.cc"),
    "Windows user path": re.compile(r"(?i)C:\\Users\\"),
    "server user path": re.compile(r"/home/crq"),
    "IPv4 literal": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "credential assignment": re.compile(
        r"(?im)^\s*(?:export\s+)?(?:[A-Z_]*(?:API_KEY|TOKEN|PASSWORD|SECRET|PRIVATE_KEY)[A-Z_]*)\s*[:=]\s*[^<\s]"
    ),
}
TEXT_EXTENSIONS = {".md", ".py", ".json", ".yaml", ".yml", ".txt", ".sh", ".gitignore"}


def main() -> int:
    violations: list[str] = []
    count = 0
    total = 0
    for path in REPO.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(REPO).as_posix()
        count += 1
        total += path.stat().st_size
        if path.name in BAD_NAMES or path.suffix.lower() in BAD_EXTENSIONS:
            violations.append(f"forbidden file: {rel}")
        if path.stat().st_size > 100_000_000:
            violations.append(f"over 100 MB: {rel}")
        if path.suffix.lower() in TEXT_EXTENSIONS or path.name == ".gitignore":
            content = path.read_text(encoding="utf-8", errors="replace")
            if path.name == "audit_public.py":
                continue  # Its regex definitions intentionally contain the search patterns.
            for label, pattern in BAD_TEXT.items():
                if pattern.search(content):
                    violations.append(f"{label}: {rel}")
    if violations:
        print("PUBLIC AUDIT FAILED")
        for violation in violations:
            print(violation)
        return 1
    print(f"PUBLIC AUDIT PASS: {count} files, {total / 1_000_000:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
