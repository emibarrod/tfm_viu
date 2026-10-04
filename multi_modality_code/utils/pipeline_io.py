"""Shared input-resolution helper for the staged multimodal sepsis pipeline.

Each pipeline stage writes only to its own folder, but downstream stages read
files produced by several upstream stages. ``resolve_input`` locates a file by
name across an ordered list of stage directories, with the more-processed
directory listed first so that processed variants (for example
``demog_processed.csv`` or the augmented ``fluid.csv``) win over their raw
extraction counterparts.
"""

import os


def resolve_input(filename: str, search_dirs) -> str:
    """Return the first existing path for ``filename`` across ``search_dirs``.

    Directories are checked in order, so callers should list the most-processed
    stage directory first. Raises ``FileNotFoundError`` if no match is found.
    """
    for directory in search_dirs:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{filename} not found in {list(search_dirs)}")
