"""Synthetic test: 2 objects moving in opposite directions, one with 2 missed frames. IDs must stay stable."""
import numpy as np
from sort_tracker import Sort

def box(cx, cy, w=40, h=60): return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

rng = np.random.default_rng(0)
t = Sort(max_age=10, min_hits=3)
ids_a, ids_b = set(), set()
for f in range(60):
    a = (50 + 8*f, 100 + 2*f)          # moves right
    b = (500 - 8*f, 300 - 1*f)         # moves left
    dets = []
    for k, (cx, cy) in enumerate([a, b]):
        if f in (25, 26) and k == 0:    # object A missed for 2 frames
            continue
        dets.append(box(cx + rng.normal(0, 1), cy + rng.normal(0, 1)) + [k])
    for tid, bx, c in t.update(np.array(dets)):
        (ids_a if c == 0 else ids_b).add(tid)
print("ids for A:", ids_a, "ids for B:", ids_b)
assert len(ids_a) == 1 and len(ids_b) == 1, "ID switch or fragmentation"
print("PASS")
