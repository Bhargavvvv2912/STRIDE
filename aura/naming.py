"""Mapping between Python import names and PyPI distribution names.

``import cv2`` comes from the distribution ``opencv-python``; ``import sklearn``
from ``scikit-learn``; ``import PIL`` from ``pillow``. The previous code keyed
usage scores by import name and then looked them up by distribution name, so the
score read 0 for exactly the heavyweight libraries the prioritisation is meant to
rank first.

The authoritative mapping lives in the installed environment's package metadata
(``importlib.metadata.packages_distributions``), so we read it from the target
interpreter when one is available and fall back to a small static table -- used
only for well-known packages whose metadata we cannot query -- otherwise.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache

from packaging.utils import canonicalize_name

# Consulted only when environment metadata is unavailable. Deliberately short:
# the dynamic path is the real mechanism, this is a safety net for pre-install
# analysis. Keys are top-level import names, values are distribution names.
STATIC_FALLBACK: dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "OpenSSL": "pyopenssl",
    "Crypto": "pycryptodome",
    "serial": "pyserial",
    "usb": "pyusb",
    "google": "protobuf",
    "pkg_resources": "setuptools",
    "mpl_toolkits": "matplotlib",
    "tensorboard": "tensorboard",
    "lxml": "lxml",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "fitz": "pymupdf",
    "jwt": "pyjwt",
    "zoneinfo": "backports-zoneinfo",
    "importlib_metadata": "importlib-metadata",
    "pydantic_core": "pydantic-core",
    "torchvision": "torchvision",
    "llvmlite": "llvmlite",
}

_PROBE = (
    "import json,sys;"
    "from importlib.metadata import packages_distributions;"
    "print(json.dumps(packages_distributions()))"
)


def canonical(name: str) -> str:
    """PEP 503 canonical form: lowercased, with ``_`` and ``.`` folded to ``-``."""
    return canonicalize_name(name)


@lru_cache(maxsize=16)
def _probe_environment(python_executable: str) -> tuple[tuple[str, str], ...]:
    try:
        result = subprocess.run(
            [python_executable, "-c", _PROBE],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return ()
        raw: dict[str, list[str]] = json.loads(result.stdout)
    except Exception:
        return ()
    # A top-level module can be provided by several distributions (namespace
    # packages). We keep the first, which importlib orders by install layout.
    return tuple((mod, canonical(dists[0])) for mod, dists in raw.items() if dists)


def import_to_distribution(
    import_name: str, python_executable: str | None = None
) -> str:
    """Best-effort distribution name (canonical) for a top-level import name."""
    top = import_name.split(".")[0]
    if python_executable:
        for mod, dist in _probe_environment(python_executable):
            if mod == top:
                return dist
    if top in STATIC_FALLBACK:
        return canonical(STATIC_FALLBACK[top])
    return canonical(top)


def build_mapping(
    import_names: set[str], python_executable: str | None = None
) -> dict[str, str]:
    """Map each top-level import name to its canonical distribution name."""
    return {name: import_to_distribution(name, python_executable) for name in import_names}
