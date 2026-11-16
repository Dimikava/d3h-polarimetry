"""
Prototype utilities for testing POI-driven region masks.

This module is intentionally separate from the production twist-angle workflow.
It supports:
- signal-shape template classification across the polarization stack
- seeded intensity-drop region growing on a reference intensity image
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class PrototypeConfig:
    method: str = "intensity"
    reference_mode: str = "mean"
    patch_radius: int = 1
    min_similarity: float = 0.92
    min_margin: float = 0.03
    background_fraction: float = 0.15
    intensity_thresholds: Optional[Tuple[float, ...]] = None


def extract_patch_mean(
    image: np.ndarray, r: int, c: int, patch_radius: int
) -> float:
    """Average a square local patch on a 2D image."""
    h, w = image.shape
    r1 = max(0, r - patch_radius)
    r2 = min(h, r + patch_radius + 1)
    c1 = max(0, c - patch_radius)
    c2 = min(w, c + patch_radius + 1)
    return float(image[r1:r2, c1:c2].mean())


def _neighbors8(r: int, c: int, h: int, w: int) -> Sequence[Tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield (rr, cc)


def keep_seed_component(mask: np.ndarray, seed: Tuple[int, int]) -> np.ndarray:
    """Keep only the connected component containing the seed."""
    h, w = mask.shape
    sr, sc = seed
    if not (0 <= sr < h and 0 <= sc < w) or not mask[sr, sc]:
        return np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)
    q: deque[Tuple[int, int]] = deque([(sr, sc)])
    kept[sr, sc] = True
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors8(r, c, h, w):
            if mask[rr, cc] and (not kept[rr, cc]):
                kept[rr, cc] = True
                q.append((rr, cc))
    return kept


def grow_intensity_masks(
    reference_image: np.ndarray,
    poi_points: Sequence[Tuple[int, int]],
    config: PrototypeConfig,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]
]:
    """Connected bright-region extraction with per-POI intensity thresholds."""
    image = reference_image.astype(float)
    h, w = image.shape
    seed_levels = np.array(
        [
            extract_patch_mean(image, r, c, config.patch_radius)
            for r, c in poi_points
        ],
        dtype=float,
    )
    positive_seed_levels = seed_levels[seed_levels > 0]
    if positive_seed_levels.size > 0:
        background_floor = config.background_fraction * float(
            np.min(positive_seed_levels)
        )
    else:
        background_floor = np.inf
    assignment_map = np.full((h, w), -1, dtype=int)
    best_score_map = np.full((h, w), np.nan, dtype=float)
    margin_map = np.full((h, w), np.nan, dtype=float)
    score_stack = np.full((len(poi_points), h, w), np.nan, dtype=float)
    masks: List[np.ndarray] = []
    thresholds = config.intensity_thresholds or ()
    for poi_idx, ((sr, sc), seed_level) in enumerate(
        zip(poi_points, seed_levels)
    ):
        if not (0 <= sr < h and 0 <= sc < w) or seed_level <= 0:
            masks.append(np.zeros((h, w), dtype=bool))
            continue
        if image[sr, sc] < background_floor:
            masks.append(np.zeros((h, w), dtype=bool))
            continue
        threshold_fraction = (
            thresholds[poi_idx] if poi_idx < len(thresholds) else 0.7
        )
        threshold_value = max(
            background_floor, threshold_fraction * seed_level
        )
        mask = np.zeros((h, w), dtype=bool)
        q: deque[Tuple[int, int]] = deque([(sr, sc)])
        mask[sr, sc] = True
        while q:
            r, c = q.popleft()
            for rr, cc in _neighbors8(r, c, h, w):
                if mask[rr, cc]:
                    continue
                if image[rr, cc] < threshold_value:
                    continue
                mask[rr, cc] = True
                q.append((rr, cc))
        mask = keep_seed_component(mask, (sr, sc))
        masks.append(mask)
        denom = max(seed_level - background_floor, 1e-12)
        score_stack[poi_idx, mask] = np.clip(
            (image[mask] - background_floor) / denom, 0.0, None
        )
    if len(poi_points) == 0:
        return (assignment_map, best_score_map, margin_map, [], [])
    valid_stack = np.where(np.isfinite(score_stack), score_stack, -np.inf)
    best_val = np.max(valid_stack, axis=0)
    if len(poi_points) > 1:
        second_val = np.partition(valid_stack, -2, axis=0)[-2]
    else:
        second_val = np.full((h, w), np.nan, dtype=float)
    for r in range(h):
        for c in range(w):
            owners = [idx for idx, mask in enumerate(masks) if mask[r, c]]
            if not owners or image[r, c] < background_floor:
                continue
            if len(owners) == 1:
                chosen = owners[0]
            else:
                distances = np.array(
                    [
                        (r - poi_points[idx][0]) ** 2
                        + (c - poi_points[idx][1]) ** 2
                        for idx in owners
                    ],
                    dtype=float,
                )
                chosen = owners[int(np.argmin(distances))]
            assignment_map[r, c] = chosen
            best_score_map[r, c] = valid_stack[chosen, r, c]
            owner_scores = np.array(
                [valid_stack[idx, r, c] for idx in owners], dtype=float
            )
            finite_scores = owner_scores[
                np.isfinite(owner_scores) & (owner_scores > -np.inf / 2)
            ]
            if finite_scores.size >= 2:
                top_two = np.partition(finite_scores, -2)[-2:]
                margin_map[r, c] = float(top_two[1] - top_two[0])
    for poi_idx, (sr, sc) in enumerate(poi_points):
        if 0 <= sr < h and 0 <= sc < w:
            assignment_map[sr, sc] = poi_idx
            best_score_map[sr, sc] = valid_stack[poi_idx, sr, sc]
    cleaned_assignment = np.full_like(assignment_map, -1)
    final_masks: List[np.ndarray] = []
    for poi_idx, seed in enumerate(poi_points):
        mask = assignment_map == poi_idx
        mask = keep_seed_component(mask, seed)
        final_masks.append(mask)
        cleaned_assignment[mask] = poi_idx
    best_score_clean = np.full_like(best_score_map, np.nan, dtype=float)
    margin_clean = np.full_like(margin_map, np.nan, dtype=float)
    for mask in final_masks:
        best_score_clean[mask] = best_score_map[mask]
        margin_clean[mask] = margin_map[mask]
    return (
        cleaned_assignment,
        best_score_clean,
        margin_clean,
        final_masks,
        [],
    )
