"""CPU baseline: quadrilaterals -> perspective warp -> visual shortlist -> ORB geometry."""
import json
from pathlib import Path
import cv2
import numpy as np
from catalog import ROOT, DATA

# Avoid oversubscribing a desktop CPU for many small matching operations.
cv2.setNumThreads(2)

SIZE = (252, 352)

def order(points):
    pts = np.asarray(points, np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]
    pts = np.roll(pts, -np.argmin(pts.sum(axis=1)), axis=0)
    # Canonical portrait even for tapped/landscape cards.
    if np.linalg.norm(pts[1]-pts[0]) > np.linalg.norm(pts[2]-pts[1]):
        pts = np.roll(pts, -1, axis=0)
    return pts

def warp(frame, points):
    dest = np.float32([[0, 0], [251, 0], [251, 351], [0, 351]])
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(order(points), dest), SIZE)

def detect(frame, min_area=2200):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    total = frame.shape[0]*frame.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > total * .85:
            continue
        poly = cv2.approxPolyDP(c, .025*cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        pts = order(poly)
        width = (np.linalg.norm(pts[1]-pts[0])+np.linalg.norm(pts[2]-pts[3]))/2
        height = (np.linalg.norm(pts[2]-pts[1])+np.linalg.norm(pts[3]-pts[0]))/2
        if not .48 < width/max(height, 1) < .9:
            continue
        candidates.append((area, pts))
    result = []
    for _, pts in sorted(candidates, key=lambda v: -v[0]):
        if any(cv2.pointPolygonTest(q, tuple(map(float, pts.mean(axis=0))), False) >= 0 for q in result):
            continue
        result.append(pts)
    return result[:16]

def signature(image):
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
    vec = cv2.dct(small)[:12, :12].flatten()[1:]
    vec -= vec.mean()
    return vec / (np.linalg.norm(vec)+1e-8)

class Recognizer:
    def __init__(self):
        path = DATA / 'cards.json'
        self.cards = json.loads(path.read_text('utf-8')) if path.exists() else {}
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.refs = {}
        self.keys, vectors = [], []
        for key, card in self.cards.items():
            image = cv2.imread(str(ROOT / card['image']))
            if image is None:
                continue
            image = cv2.resize(image, SIZE)
            self.keys.append(key)
            vectors.append(signature(image))
        self.vectors = np.array(vectors, np.float32)
        # Small galleries can be warmed once rather than pausing the first frame.
        if len(self.keys) <= 200:
            for key in self.keys:
                self.reference(key)

    def reference(self, key):
        if key not in self.refs:
            image = cv2.resize(cv2.imread(str(ROOT / self.cards[key]['image'])), SIZE)
            self.refs[key] = self.orb.detectAndCompute(image, None)
        return self.refs[key]

    def identify(self, crop):
        if not self.keys:
            return None, 0, 0
        scores = {}
        for rotated in [crop, cv2.rotate(crop, cv2.ROTATE_180)]:
            kp, desc = self.orb.detectAndCompute(rotated, None)
            if desc is None or len(desc) < 8:
                continue
            similarity = self.vectors @ signature(rotated)
            shortlist = np.argsort(similarity)[-min(12, len(self.keys)):]
            for idx in shortlist:
                key = self.keys[idx]
                rk, rd = self.reference(key)
                if rd is None or len(rd) < 2:
                    continue
                pairs = self.matcher.knnMatch(desc, rd, k=2)
                good = [m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < .72*n.distance]
                if len(good) < 8:
                    continue
                src = np.float32([kp[m.queryIdx].pt for m in good])
                dst = np.float32([rk[m.trainIdx].pt for m in good])
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4)
                inliers = int(mask.sum()) if mask is not None else 0
                scores[key] = max(scores.get(key, 0), inliers)
        ranked = sorted(scores.items(), key=lambda v: -v[1])
        if not ranked:
            return None, 0, 0
        key, best = ranked[0]
        second = ranked[1][1] if len(ranked)>1 else 0
        accepted = best >= 14 and best-second >= 4
        return key if accepted else None, best, best-second
