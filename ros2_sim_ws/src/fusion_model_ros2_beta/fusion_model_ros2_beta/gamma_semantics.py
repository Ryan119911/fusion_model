"""Convert between model-local brush twist and ROS absolute tool heading."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def wrap(values):
    values = np.asarray(values, dtype=float)
    return np.arctan2(np.sin(values), np.cos(values))


def forward_headings(xy, stroke_ids):
    """Match PaperFusionRenderer.forward_trajectory_heading at source points."""
    xy = np.asarray(xy, dtype=float)
    stroke_ids = np.asarray(stroke_ids)
    if xy.ndim != 2 or xy.shape[1] != 2 or stroke_ids.shape != (len(xy),):
        raise ValueError("invalid XY/stroke arrays")
    result = np.zeros(len(xy), dtype=float)
    for stroke in dict.fromkeys(stroke_ids.tolist()):
        indices = np.flatnonzero(stroke_ids == stroke)
        if len(indices) < 2:
            result[indices] = 0.0
            continue
        delta = xy[indices[1:]] - xy[indices[:-1]]
        heading = np.arctan2(delta[:, 1], delta[:, 0])
        result[indices[:-1]] = heading
        result[indices[-1]] = heading[-1]
    return result


def absolute_to_local(gamma, xy, stroke_ids):
    return wrap(np.asarray(gamma, dtype=float) - forward_headings(xy, stroke_ids))


def local_to_absolute(gamma, xy, stroke_ids):
    return wrap(np.asarray(gamma, dtype=float) + forward_headings(xy, stroke_ids))


def convert_physical_csv_local_to_absolute(path: Path) -> dict:
    """Convert a contact-only physical CSV in place and return an audit."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames or ()), list(reader)
    xy = np.asarray([[float(row["x"]), float(row["y"])] for row in rows])
    strokes = np.asarray([int(float(row["stroke_id"])) for row in rows])
    local = np.asarray([float(row["gamma"]) for row in rows])
    absolute = local_to_absolute(local, xy, strokes)
    for row, value in zip(rows, absolute):
        row["gamma"] = repr(float(value))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "model_gamma": "local_rotation_relative_to_forward_xy_heading",
        "ros_gamma": "absolute_forward_heading_plus_local_rotation",
        "local_limit_rad": [-math.pi / 6.0, math.pi / 6.0],
        "local_range_rad": [float(local.min()), float(local.max())],
        "absolute_range_rad": [float(absolute.min()), float(absolute.max())],
    }
