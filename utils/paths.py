import os
import subprocess
from pathlib import Path

def is_wsl() -> bool:
    """Detect WSL environment."""
    return os.name == "posix" and (
        "WSL_DISTRO_NAME" in os.environ
        or "microsoft" in os.uname().release.lower()
    )

def normalize_path(path_str: str) -> str:
    """
    Normalize paths so Windows-style paths work under WSL/Linux.
    If on WSL and the path looks Windows-y, convert with wslpath (fallback to simple slash replace).
    Otherwise, return unchanged.
    """
    if not path_str:
        return path_str
    if is_wsl() and (":" in path_str or "\\" in path_str):
        try:
            return subprocess.check_output(["wslpath", "-u", path_str], text=True).strip()
        except Exception:
            return path_str.replace("\\", "/")
    return path_str

def ensure_path(p: str | Path) -> Path:
    """Convenience to coerce to Path after normalization."""
    return Path(normalize_path(str(p)))
