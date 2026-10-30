"""Minimal SORT (Bewley et al. 2016): constant-velocity Kalman filter per track,
Hungarian assignment on IoU. Boxes are [x1, y1, x2, y2]."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def to_z(box):  # box -> [cx, cy, area, aspect]
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)])


def to_box(x):
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r); h = s / w
    return np.array([x[0] - w / 2, y := x[1] - h / 2, x[0] + w / 2, y + h])


class Track:
    _next_id = 1

    def __init__(self, box, cls):
        self.F = np.eye(7); self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.H = np.eye(4, 7)
        self.R = np.diag([1, 1, 10, 10]).astype(float)
        self.Q = np.diag([1, 1, 1, 1, .01, .01, .0001]).astype(float)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(float)
        self.x = np.zeros(7); self.x[:4] = to_z(box)
        self.cls, self.hits, self.since_update, self.age = cls, 1, 0, 0
        self.id = Track._next_id; Track._next_id += 1

    def predict(self):
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1; self.since_update += 1
        return to_box(self.x)

    def update(self, box, cls):
        y = to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.cls, self.hits, self.since_update = cls, self.hits + 1, 0


class Sort:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_threshold
        self.tracks, self.frame = [], 0

    def update(self, dets):
        """dets: array (N, 5+) of [x1,y1,x2,y2,cls]. Returns list of (id, box, cls)."""
        self.frame += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        ok = ~np.isnan(preds).any(axis=1)
        self.tracks = [t for t, k in zip(self.tracks, ok) if k]; preds = preds[ok]

        matched_d, matched_t = set(), set()
        if len(self.tracks) and len(dets):
            iou = iou_matrix(preds, dets[:, :4])
            r, c = linear_sum_assignment(-iou)
            for ti, di in zip(r, c):
                if iou[ti, di] >= self.iou_thr:
                    self.tracks[ti].update(dets[di, :4], int(dets[di, 4]))
                    matched_t.add(ti); matched_d.add(di)
        for di in range(len(dets)):
            if di not in matched_d:
                self.tracks.append(Track(dets[di, :4], int(dets[di, 4])))

        out = []
        for t in self.tracks:
            if t.since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits):
                out.append((t.id, to_box(t.x), t.cls))
        self.tracks = [t for t in self.tracks if t.since_update <= self.max_age]
        return out
