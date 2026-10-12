"""Repository-relative resource paths for script entry points.

Returns strings for compatibility with existing os.path and string-based code.
This module does not create directories, read data or execute analysis.
"""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def repo_path(relative=""):
    """Resolve a repository-owned resource relative to this checkout."""
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("repo_path expects a repository-relative path.")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("Resource path escapes the repository.") from exc
    return str(resolved)