"""Structural type for a layout block.

`postprocess` imports the NID extractor and the NID extractor needs the shape of
a layout block, so the shared type lives here to keep that dependency one-way.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LayoutBlockLike(Protocol):
    category: str
    text: str
    bbox: tuple[float, float, float, float] | None
