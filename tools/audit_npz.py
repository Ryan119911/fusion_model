import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.feature_schema import read_npz_schema


def npy_header(archive: zipfile.ZipFile, member: str):
    with archive.open(member) as file:
        version = np.lib.format.read_magic(file)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = reader(file)
    return {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran}


def main(args):
    path = Path(args.npz_path)
    with zipfile.ZipFile(path) as archive:
        arrays = {
            name[:-4]: npy_header(archive, name)
            for name in archive.namelist()
            if name.endswith(".npy")
        }
    data = np.load(path, allow_pickle=True)
    inputs = np.asarray(data["inputs"])
    meta = np.asarray(data["meta"], dtype=object) if "meta" in data.files else []
    schema = read_npz_schema(data, inputs.shape[1])
    groups = Counter()
    real = 0
    for value in meta:
        item = value if isinstance(value, dict) else value.item()
        groups[str(item.get("sample_id"))] += 1
        real += int(bool(item.get("used_real_image")))
    report = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "arrays": arrays,
        "feature_schema": schema,
        "input_finite": bool(np.isfinite(inputs).all()),
        "input_min": inputs.min(axis=0).tolist(),
        "input_max": inputs.max(axis=0).tolist(),
        "groups": len(groups),
        "real_targets": real,
        "synthetic_targets": len(meta) - real,
    }
    data.close()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--output")
    main(parser.parse_args())
