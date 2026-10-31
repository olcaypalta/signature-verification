"""Generate synthetic signature-like images for a quick demo.

A "writer" is a smooth random pen trajectory (sum of sinusoids). Genuine samples add
small natural variation (scale, slant, timing, jitter); a skilled-forgery sample keeps the
coarse trajectory but changes its fine structure; a random forgery is another writer.
Real evaluations should use CEDAR, GPDS or MCYT signature datasets.
"""
import os

import cv2
import numpy as np


def trajectory(seed, n=600):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    x = t * 700 + sum(rng.uniform(5, 30) * np.sin(2 * np.pi * rng.uniform(3, 14) * t + rng.uniform(0, 6))
                      for _ in range(3))
    y = sum(rng.uniform(15, 55) * np.sin(2 * np.pi * rng.uniform(2, 12) * t + rng.uniform(0, 6))
            for _ in range(4))
    return np.stack([x, y], 1)


def render(traj, seed, variation=1.0):
    rng = np.random.default_rng(seed)
    p = traj.copy()
    p[:, 0] *= 1 + rng.normal(0, 0.04 * variation)
    p[:, 1] *= 1 + rng.normal(0, 0.06 * variation)
    p[:, 0] += p[:, 1] * rng.normal(0, 0.08 * variation)               # slant
    p += cv2.GaussianBlur(rng.normal(0, 3 * variation, p.shape), (1, 31), 0)  # smooth jitter
    p -= p.min(0)
    p += 30
    img = np.full((int(p[:, 1].max()) + 30, int(p[:, 0].max()) + 30), 255, np.uint8)
    cv2.polylines(img, [p.astype(np.int32)], False, 20, thickness=int(rng.integers(2, 4)), lineType=cv2.LINE_AA)
    return img


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
    gen_dir, q_dir = os.path.join(root, "genuine"), os.path.join(root, "questioned")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(q_dir, exist_ok=True)
    writer = trajectory(7)
    for i in range(5):
        cv2.imwrite(os.path.join(gen_dir, f"ref_{i + 1}.png"), render(writer, 100 + i))
    cv2.imwrite(os.path.join(q_dir, "q1_genuine.png"), render(writer, 200))
    cv2.imwrite(os.path.join(q_dir, "q2_genuine.png"), render(writer, 201))
    forged = writer + np.stack([np.zeros(len(writer)), 18 * np.sin(np.linspace(0, 40, len(writer)))], 1)
    cv2.imwrite(os.path.join(q_dir, "q3_skilled_forgery.png"), render(forged, 300))
    cv2.imwrite(os.path.join(q_dir, "q4_random_forgery.png"), render(trajectory(99), 301))
    print("saved 5 references to examples/genuine and 4 questioned samples to examples/questioned")
