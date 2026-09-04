"""Frame-to-frame change analysis."""
from __future__ import annotations

from dataclasses import dataclass
from .observation import Grid
from .perception import FrameAbstraction, ObjectDescription, segment


@dataclass(frozen=True)
class FrameDiff:
    changed_cells: tuple[tuple[int, int, int | None, int | None], ...]
    changed_bbox: tuple[int, int, int, int] | None
    disappeared_objects: tuple[str, ...]
    appeared_objects: tuple[str, ...]
    moved_objects: tuple[tuple[str, str], ...]
    recolored_objects: tuple[tuple[str, int, int], ...]
    resized_objects: tuple[tuple[str, int, int], ...]
    unchanged_cells: int
    no_visible_effect: bool

    def summary(self) -> str:
        if self.no_visible_effect: return "no visible effect"
        bits = [f"{len(self.changed_cells)} cells changed"]
        if self.moved_objects: bits.append(f"{len(self.moved_objects)} object(s) moved")
        if self.recolored_objects: bits.append(f"{len(self.recolored_objects)} recolored")
        if self.resized_objects: bits.append(f"{len(self.resized_objects)} resized")
        if self.appeared_objects: bits.append(f"{len(self.appeared_objects)} appeared")
        if self.disappeared_objects: bits.append(f"{len(self.disappeared_objects)} disappeared")
        return "; ".join(bits)


def _match(old: FrameAbstraction, new: FrameAbstraction) -> tuple[dict[str,ObjectDescription], set[str], set[str]]:
    unmatched_new = set(range(len(new.objects))); pairs: dict[str,ObjectDescription] = {}
    for o in old.objects:
        candidates = [(i,n) for i,n in enumerate(new.objects) if i in unmatched_new and n.color == o.color and (set(o.shape_signature) == set(n.shape_signature) or abs(o.area-n.area)<=1)]
        if not candidates:
            # A recolored object can still be matched by shape/position.
            candidates = [(i,n) for i,n in enumerate(new.objects) if i in unmatched_new and (set(o.shape_signature) == set(n.shape_signature) and abs(o.centroid[0]-n.centroid[0]) <= 2 and abs(o.centroid[1]-n.centroid[1]) <= 2)]
        if candidates:
            i,n = min(candidates, key=lambda p: abs(p[1].centroid[0]-o.centroid[0])+abs(p[1].centroid[1]-o.centroid[1]))
            pairs[o.id]=n; unmatched_new.remove(i)
    return pairs, {o.id for o in old.objects if o.id not in pairs}, {new.objects[i].id for i in unmatched_new}


def frame_diff(previous: Grid | None, current: Grid) -> FrameDiff:
    if previous is None:
        n = sum(len(r) for r in current)
        return FrameDiff(tuple(), None, tuple(), tuple(o.id for o in segment(current).objects), tuple(), tuple(), tuple(), n, True)
    h=max(len(previous),len(current)); w=max(len(previous[0]) if previous else 0,len(current[0]) if current else 0)
    changes=[]
    for y in range(h):
        for x in range(w):
            a = previous[y][x] if y<len(previous) and x<len(previous[y]) else None
            b = current[y][x] if y<len(current) and x<len(current[y]) else None
            if a != b: changes.append((y,x,a,b))
    old_abs,new_abs=segment(previous),segment(current); pairs,dis,app=_match(old_abs,new_abs)
    moved=[]; recol=[]; resized=[]
    old_by={o.id:o for o in old_abs.objects}
    for oid,n in pairs.items():
        o=old_by[oid]
        if o.centroid != n.centroid: moved.append((oid,n.id))
        if o.color != n.color: recol.append((oid,o.color,n.color))
        if o.area != n.area: resized.append((oid,o.area,n.area))
    bbox=None
    if changes: bbox=(min(c[0] for c in changes),min(c[1] for c in changes),max(c[0] for c in changes),max(c[1] for c in changes))
    return FrameDiff(tuple(changes),bbox,tuple(sorted(dis)),tuple(sorted(app)),tuple(moved),tuple(recol),tuple(resized),h*w-len(changes),not changes)
