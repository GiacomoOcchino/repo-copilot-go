from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class FileHunk:
    path: str
    # lista di (start_line_new, line_count_new) dalla parte "b/" del diff
    hunks: List[Tuple[int, int]]


def get_git_diff(range_spec: str) -> str:
    """
    Esegue `git diff <range_spec>` e restituisce testo diff.
    range_spec esempio: "HEAD~1..HEAD"
    """
    res = subprocess.run(
        ["git", "diff", range_spec],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        shell=False,
        check=False,
    )
    if res.returncode != 0 and not res.stdout:
        raise RuntimeError(f"git diff failed: {res.stderr.strip()}")
    return res.stdout


def parse_unified_diff_files(diff_text: str) -> List[FileHunk]:
    """
    Estrae file cambiati e hunk line numbers (lato 'b/..', cioè file risultante).
    """
    files: Dict[str, List[Tuple[int, int]]] = {}
    current_file: str | None = None

    diff_lines = diff_text.splitlines()

    # esempio:
    # diff --git a/src/x.py b/src/x.py
    file_re = re.compile(r"^diff --git a/(.+?) b/(.+)$")
    # esempio hunk:
    # @@ -10,7 +10,9 @@
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for ln in diff_lines:
        m = file_re.match(ln)
        if m:
            # prendiamo il path lato b/
            current_file = m.group(2).strip()
            files.setdefault(current_file, [])
            continue

        if current_file:
            mh = hunk_re.match(ln)
            if mh:
                start = int(mh.group(1))
                count = int(mh.group(2) or "1")
                files[current_file].append((start, count))

    return [FileHunk(path=k, hunks=v) for k, v in files.items()]


def extract_file_context(
    repo_root: Path, fh: FileHunk, pad: int = 6, max_chars: int = 2000
) -> str:
    """
    Legge il file corrente e ritorna snippet intorno alle linee toccate.
    Se il file non esiste (deleted/renamed), ritorna stringa vuota.
    """
    p = (repo_root / fh.path).resolve()
    if not p.exists() or not p.is_file():
        return ""

    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    spans: List[Tuple[int, int]] = []
    for start, count in fh.hunks:
        a = max(1, start - pad)
        b = min(len(lines), start + count + pad)
        spans.append((a, b))

    # merge spans sovrapposti
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for a, b in spans:
        if not merged or a > merged[-1][1] + 1:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    chunks: List[str] = []
    for a, b in merged:
        snippet = "\n".join(lines[i - 1] for i in range(a, b + 1))
        chunks.append(f"# {fh.path} lines {a}-{b}\n{snippet}")

    out = "\n...\n".join(chunks)
    return out[:max_chars]
