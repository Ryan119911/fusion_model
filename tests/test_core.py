import json
import tempfile
import unittest
from pathlib import Path

from models.dynamic_brush import DynamicBrushModel, DynamicBrushParams, PolyFit1D
from models.geometry import resample_stroke
from utils.feature_schema import STROKE10_V1, get_feature_schema
from utils.splits import build_split
from utils.types import StrokeTrajectory, TrajectoryPoint


class CoreTests(unittest.TestCase):
    def test_group_split_has_no_leakage(self):
        meta = [{"sample_id": index // 3} for index in range(30)]
        train, val, manifest = build_split(meta, len(meta), 0.2, 42)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(manifest["train_groups"]) & set(manifest["val_groups"]))

    def test_schema_dimension(self):
        self.assertEqual(get_feature_schema(STROKE10_V1).input_dim, 10)

    def test_resample_preserves_endpoints(self):
        stroke = StrokeTrajectory(0, [
            TrajectoryPoint(0, 0, 0, 0, 0, 0, 0, 0, 1),
            TrajectoryPoint(0, 1, 10, 0, 4, 0, 0, 0, 2),
        ])
        result = resample_stroke(stroke, 5).sorted_points()
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0].x, 0)
        self.assertAlmostEqual(result[-1].x, 10)
        self.assertAlmostEqual(result[-1].z, 4)

    def test_dynamic_model_json_roundtrip(self):
        model = DynamicBrushModel(DynamicBrushParams(
            mode="calibrated",
            width_fn=PolyFit1D([1, 2]),
            drag_fn=PolyFit1D([0, 3]),
            offset_fn=PolyFit1D([0.5]),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save_json(str(path))
            loaded = DynamicBrushModel.load_json(str(path))
            self.assertAlmostEqual(loaded.width(2), 5)
            self.assertEqual(json.loads(path.read_text())["mode"], "calibrated")


if __name__ == "__main__":
    unittest.main()
