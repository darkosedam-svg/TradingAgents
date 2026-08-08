"""Versioned, content-hashed prompts.

Every decision logs the SHA of the prompt that produced it, so an eval
regression is traceable to a specific edit rather than to "something changed
last week". Prompt files are named ``<task>.v<N>.md``; bumping ``N`` is how you
change a prompt, and the loader always serves the highest version present unless
one is pinned explicitly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent
_FILENAME = re.compile(r"^(?P<task>[a-z0-9_]+)\.v(?P<version>\d+)\.md$")


@dataclass(frozen=True)
class Prompt:
    """A system preamble plus its identity."""

    task: str
    version: int
    text: str
    sha256: str

    @property
    def ref(self) -> str:
        """Stable, loggable identifier: ``sentiment.v1@a1b2c3d4``."""
        return f"{self.task}.v{self.version}@{self.sha256[:8]}"


def available_versions(task: str) -> list[int]:
    versions = []
    for path in PROMPT_DIR.glob(f"{task}.v*.md"):
        match = _FILENAME.match(path.name)
        if match:
            versions.append(int(match.group("version")))
    return sorted(versions)


@lru_cache(maxsize=None)
def load(task: str, version: int | None = None) -> Prompt:
    """Load a prompt by task name, defaulting to the highest version on disk."""
    versions = available_versions(task)
    if not versions:
        raise FileNotFoundError(f"no prompt files for task {task!r} in {PROMPT_DIR}")

    resolved = versions[-1] if version is None else version
    if resolved not in versions:
        raise FileNotFoundError(
            f"prompt {task}.v{resolved} not found; have versions {versions}"
        )

    path = PROMPT_DIR / f"{task}.v{resolved}.md"
    text = path.read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Prompt(task=task, version=resolved, text=text, sha256=digest)


def registry() -> dict[str, Prompt]:
    """Every task's current prompt, for boot-time logging and drift reports."""
    tasks = set()
    for path in PROMPT_DIR.glob("*.v*.md"):
        match = _FILENAME.match(path.name)
        if match:
            tasks.add(match.group("task"))
    return {task: load(task) for task in sorted(tasks)}


__all__ = ["Prompt", "available_versions", "load", "registry", "PROMPT_DIR"]
