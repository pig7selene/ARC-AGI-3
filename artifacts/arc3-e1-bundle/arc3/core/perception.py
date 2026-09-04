"""Generic, game-independent grid perception."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, deque
from typing import Iterable

from .observation import Grid


@dataclass(frozen=True)
class ObjectDescription:
    id: str
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    shape_signature: tuple[str, ...]
    border_contact: tuple[str, ...]
    area: int
    holes: int = 0
    adjacent_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameAbstraction:
    height: int
    width: int
    color_frequencies: dict[int, int]
    objects: tuple[ObjectDescription, ...]
    symmetry: tuple[str, ...] = ()


def _shape_signature(cells: set[tuple[int, int]]) -> tuple[str, ...]:
    min_y = min(y for y, _ in cells); min_x = min(x for _, x in cells)
    return tuple(f"{y-min_y},{x-min_x}" for y, x in sorted(cells))


def segment(grid: Grid, background: int | None = None) -> FrameAbstraction:
    h, w = len(grid), len(grid[0]) if grid else 0
    counts = Counter(c for row in grid for c in row)
    if background is None and counts:
        background = counts.most_common(1)[0][0]
    seen: set[tuple[int, int]] = set(); objects: list[ObjectDescription] = []
    for y in range(h):
        for x in range(w):
            if (y, x) in seen or grid[y][x] == background:
                continue
            color = grid[y][x]; q = deque([(y, x)]); seen.add((y, x)); cells: set[tuple[int, int]] = set()
            while q:
                cy, cx = q.popleft(); cells.add((cy, cx))
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny,nx) not in seen and grid[ny][nx] == color:
                        seen.add((ny,nx)); q.append((ny,nx))
            min_y=min(a for a,b in cells); max_y=max(a for a,b in cells); min_x=min(b for a,b in cells); max_x=max(b for a,b in cells)
            border = tuple(s for s, ok in (("top",min_y==0),("bottom",max_y==h-1),("left",min_x==0),("right",max_x==w-1)) if ok)
            oid = f"{color}:{min_y}:{min_x}:{len(cells)}"
            objects.append(ObjectDescription(oid,color,tuple(sorted(cells)),(min_y,min_x,max_y,max_x),
                (sum(a for a,b in cells)/len(cells),sum(b for a,b in cells)/len(cells)),_shape_signature(cells),border,len(cells)))
    syms=[]
    if grid and all(grid[y][x] == grid[y][w-1-x] for y in range(h) for x in range(w)): syms.append("horizontal_reflection")
    if grid and h == w and all(grid[y][x] == grid[x][y] for y in range(h) for x in range(w)): syms.append("diagonal_reflection")
    # Populate a lightweight adjacency graph after IDs are known.
    enriched=[]
    by_cell={cell:o.id for o in objects for cell in o.cells}
    for o in objects:
        neighbors=set()
        for y,x in o.cells:
            for p in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if p in by_cell and by_cell[p] != o.id: neighbors.add(by_cell[p])
        enriched.append(ObjectDescription(o.id,o.color,o.cells,o.bbox,o.centroid,o.shape_signature,o.border_contact,o.area,o.holes,tuple(sorted(neighbors))))
    return FrameAbstraction(h,w,dict(counts),tuple(enriched),tuple(syms))
