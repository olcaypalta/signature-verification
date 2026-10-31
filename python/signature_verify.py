"""Offline signature verification (writer-dependent, few references).

A user enrolls with a few genuine signatures. A questioned signature is accepted if its
feature distance to the references is not much larger than the distances among the
references themselves.

Features: HOG (shape/stroke direction) + uniform LBP on a 4x8 grid (stroke texture)
          + simple global descriptors (aspect ratio, ink density, centroid).
Decision: d(q) = mean cosine distance to references;  accept if d(q) <= mu + k * sigma,
          where mu/sigma are the pairwise distances inside the reference set.

Usage:
  python signature_verify.py --refs ../examples/genuine --query ../examples/questioned/q1.png
  python signature_verify.py --refs my_refs/ --query test/*.png --k 2.5
"""
import argparse
import glob
import os

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

SIZE = (256, 128)  # (w, h)


def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(bw)
    if len(xs) == 0:
        raise SystemExit(f"no ink found in {path}")
    bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = bw.shape
    scale = min(SIZE[0] / w, SIZE[1] / h)
    bw = cv2.resize(bw, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(SIZE[::-1], np.uint8)
    oy, ox = (SIZE[1] - bw.shape[0]) // 2, (SIZE[0] - bw.shape[1]) // 2
    canvas[oy:oy + bw.shape[0], ox:ox + bw.shape[1]] = bw
    return canvas, w / h


def features(path):
    img, aspect = preprocess(path)
    f_hog = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    f_lbp = []
    for rows in np.array_split(lbp, 4, axis=0):
        for cell in np.array_split(rows, 8, axis=1):
            h, _ = np.histogram(cell, bins=10, range=(0, 10))
            f_lbp.append(h / (h.sum() + 1e-6))
    ink = img > 0
    ys, xs = np.nonzero(ink)
    f_glob = [aspect / 4.0, ink.mean() * 5, xs.mean() / SIZE[0], ys.mean() / SIZE[1]]
    f = np.concatenate([f_hog / (np.linalg.norm(f_hog) + 1e-6),
                        np.concatenate(f_lbp) / (np.linalg.norm(f_lbp) + 1e-6),
                        np.array(f_glob)])
    return f


def cos_dist(a, b):
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def enroll(ref_paths):
    if len(ref_paths) < 3:
        raise SystemExit("need at least 3 reference signatures")
    refs = [features(p) for p in ref_paths]
    pair = [cos_dist(refs[i], refs[j]) for i in range(len(refs)) for j in range(i + 1, len(refs))]
    return refs, float(np.mean(pair)), float(np.std(pair) + 1e-4)


def verify(refs, mu, sigma, query_path, k):
    d = float(np.mean([cos_dist(features(query_path), r) for r in refs]))
    z = (d - mu) / sigma
    return d, z, z <= k


def expand(patterns):
    out = []
    for p in patterns:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.png")) + glob.glob(os.path.join(p, "*.jpg")))
        else:
            out += sorted(glob.glob(p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", required=True, help="folder or files with genuine signatures")
    ap.add_argument("--query", nargs="+", required=True, help="questioned signature(s)")
    ap.add_argument("--k", type=float, default=2.0, help="acceptance threshold in reference std units")
    args = ap.parse_args()

    ref_paths = expand(args.refs)
    refs, mu, sigma = enroll(ref_paths)
    print(f"enrolled {len(refs)} references: intra distance mu={mu:.4f} sigma={sigma:.4f}")
    for q in expand(args.query):
        d, z, ok = verify(refs, mu, sigma, q, args.k)
        print(f"  {os.path.basename(q):24s} distance={d:.4f} z={z:+6.2f} -> {'GENUINE' if ok else 'FORGERY / different writer'}")


if __name__ == "__main__":
    main()
