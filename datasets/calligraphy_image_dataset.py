from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


def _labelme_points_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    LabelMe rectangle points:
      [[x1, y1], [x2, y2]]
    转成标准 bbox:
      (x_min, y_min, x_max, y_max)
    """
    if points is None or len(points) < 2:
        raise ValueError(f"Invalid labelme points: {points}")

    x1, y1 = points[0]
    x2, y2 = points[1]

    return (
        float(min(x1, x2)),
        float(min(y1, y2)),
        float(max(x1, x2)),
        float(max(y1, y2)),
    )


def _expand_and_clip(
    bbox: Tuple[float, float, float, float],
    w: int,
    h: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    """
    按 padding 比例向外扩展 bbox，并裁剪到图像边界。
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    x1 -= bw * padding
    x2 += bw * padding
    y1 -= bh * padding
    y2 += bh * padding

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return x1, y1, x2, y2


def _crop_to_tensor(
    img: Image.Image,
    bbox: Tuple[float, float, float, float],
    image_size: Optional[int],
    grayscale: bool,
    padding: float,
) -> torch.Tensor:
    """
    从整幅作品图中裁出单字，并转成 [C,H,W] tensor。
    """
    w, h = img.size
    left, upper, right, lower = _expand_and_clip(bbox, w, h, padding)

    crop = img.crop((left, upper, right, lower))
    crop = crop.convert("L") if grayscale else crop.convert("RGB")

    if image_size is not None:
        crop = crop.resize((image_size, image_size))

    # Convert to numpy array and normalize to [0, 1]
    arr = np.array(crop).astype(np.float32) / 255.0

    # 统一极性：
    # 如果均值很高，说明大概率是白底黑字，转成黑底白字。
    # 目标统一为：背景=0，墨迹=1。
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    tensor = torch.from_numpy(arr)

    if grayscale:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.permute(2, 0, 1)

    return tensor


class CalligraphyImageDataset(Dataset):
    """
    一个样本 = 一整幅书法作品中的一个单字 crop。

    数据结构：
      image_dir/xxx/A.jpg
      json_dir/xxx/A.json

    JSON 格式：
      LabelMe 风格：
      {
        "version": "...",
        "shapes": [
          {
            "label": "言",
            "points": [[x1, y1], [x2, y2]],
            "group_id": 0,
            "shape_type": "rectangle"
          }
        ],
        "imagePath": "A.jpg",
        "imageData": "..."
      }

    注意：
      - 本类优先使用磁盘上的 image_dir/xxx/A.jpg
      - 忽略 imageData
      - 只处理 rectangle
    """

    def __init__(
        self,
        image_dir: str,
        json_dir: str,
        image_ext: str = ".jpg",
        image_size: Optional[int] = 128,
        grayscale: bool = True,
        padding: float = 0.0,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        data_csv: Optional[str] = None,
        chirography_filter: Optional[str] = None,
    ):
        self.image_dir = Path(image_dir)
        self.json_dir = Path(json_dir)
        self.image_ext = image_ext
        self.image_size = image_size
        self.grayscale = grayscale
        self.padding = padding
        self.transform = transform
        self.data_csv = data_csv
        self.chirography_filter = chirography_filter
        self.allowed_image_rels =self._build_allowed_image_rels()

        if not self.image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")

        if not self.json_dir.exists():
            raise FileNotFoundError(f"json_dir not found: {json_dir}")

        self.index: List[Dict[str, Any]] = []
        self._build_index()

    def _build_allowed_image_rels(self) -> Optional[set]:
        """
        根据 data.csv 的 chirography 字段筛选图片。
        返回允许使用的相对图片路径集合，例如：
          {"106/10.jpg", "133/5.jpg"}
        如果没有设置 chirography_filter，则返回 None，表示不过滤。
        """
        if self.chirography_filter is None:
            return None

        if self.data_csv is None:
            print(
                f"[WARN] chirography_filter={self.chirography_filter} "
                f"but data_csv is None; style filter disabled."
            )
            return None

        csv_path = Path(self.data_csv)
        if not csv_path.exists():
            print(
                f"[WARN] data_csv not found: {self.data_csv}; "
                f"style filter disabled."
            )
            return None

        df = pd.read_csv(csv_path, engine="python")

        if "img_path" not in df.columns or "chirography" not in df.columns:
            print(
                f"[WARN] data_csv missing img_path/chirography columns; "
                f"got columns={list(df.columns)}; style filter disabled."
            )
            return None

        df = df[df["chirography"].astype(str) == str(self.chirography_filter)]

        allowed = set()
        for rel in df["img_path"].astype(str).tolist():
            rel = rel.strip()
            if not rel:
                continue

            p = Path(rel)
            if p.suffix == "" and self.image_ext:
                rel = rel + self.image_ext

            # 统一成 posix 风格，避免 106\10.jpg 这种路径差异
            allowed.add(str(Path(rel).as_posix()))

        print(
            f"[INFO] Style filter enabled: chirography={self.chirography_filter}, "
            f"allowed images={len(allowed)}"
        )

        return allowed


    def _json_to_image_path(self, json_path: Path, image_path_in_json: Optional[str] = None) -> Path:
        """
        将：
          json_dir/xxx/A.json
        映射到：
          image_dir/xxx/A.jpg

        若 JSON 中 imagePath 有值，也仅用其文件名做校正。
        """
        rel_json = json_path.relative_to(self.json_dir)
        rel_img = rel_json.with_suffix(self.image_ext)

        candidate = self.image_dir / rel_img
        if candidate.exists():
            return candidate

        # 兼容 JSON 中 imagePath，比如 "10.jpg"
        if image_path_in_json:
            candidate2 = self.image_dir / rel_json.parent / Path(image_path_in_json).name
            if candidate2.exists():
                return candidate2

        return candidate

    def _build_index(self) -> None:
        json_files = sorted(self.json_dir.rglob("*.json"))

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[跳过] JSON 读取失败: {json_path}, error={e}")
                continue

            shapes = data.get("shapes", [])
            image_path_in_json = data.get("imagePath")
            image_path = self._json_to_image_path(json_path, image_path_in_json)

            if not image_path.exists():
                print(f"[跳过] 找不到对应图片: json={json_path}, image={image_path}")
                continue

            # 按书体过滤：只保留 data.csv 中指定 chirography 的图片
            if self.allowed_image_rels is not None:
                try:
                    rel_img = image_path.relative_to(self.image_dir).as_posix()
                except ValueError:
                    rel_img = str(image_path)

                if rel_img not in self.allowed_image_rels:
                    continue

            for shape_idx, shape in enumerate(shapes):
                if shape.get("shape_type") != "rectangle":
                    continue

                label = shape.get("label")
                points = shape.get("points")
                group_id = shape.get("group_id")

                if label is None or points is None or len(points) < 2:
                    continue

                try:
                    bbox = _labelme_points_to_bbox(points)
                except Exception:
                    continue

                x1, y1, x2, y2 = bbox
                if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                    continue

                self.index.append({
                    "character": str(label),
                    "bbox": bbox,
                    "group_id": group_id,
                    "shape_index": shape_idx,
                    "image_path": image_path,
                    "json_path": json_path,
                    "file_stem": image_path.stem,
                    "folder": str(image_path.parent.name),
                    "imagePath_in_json": image_path_in_json,
                })
        print(f"[INFO] Built LabelMe character index: {len(self.index)} boxes")

    def __len__(self) -> int:
        return len(self.index)

    def _build_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        with Image.open(item["image_path"]) as img:
            w, h = img.size
            x1, y1, x2, y2 = item["bbox"]

            if x2 > w or y2 > h or x1 < 0 or y1 < 0:
                print(
                    f"[越界] {item['image_path']} "
                    f"图尺寸=({w},{h}) bbox={item['bbox']}"
                )

            image = _crop_to_tensor(
                img=img,
                bbox=item["bbox"],
                image_size=self.image_size,
                grayscale=self.grayscale,
                padding=self.padding,
            )

        return {
            "image": image,                              # 裁出的单字 [C,H,W]
            "character": item["character"],              # 单字标签
            "bbox": item["bbox"],                        # 原图中的 bbox
            "group_id": item["group_id"],                # LabelMe group_id
            "shape_index": item["shape_index"],          # 在 shapes 中的索引
            "image_path": str(item["image_path"]),       # 来源整幅作品图片
            "json_path": str(item["json_path"]),         # 来源 json 标注
            "file_stem": item["file_stem"],              # 例如 10
            "folder": item["folder"],                    # 例如 xxx
            "imagePath_in_json": item["imagePath_in_json"],
            "meta": dict(item),
        }

    def __getitem__(self, index: int):
        sample = self._build_sample(self.index[index])
        if self.transform is not None:
            return self.transform(sample)
        return sample


def collate_calligraphy_image_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)

    return {
        "images": images,
        "characters": [item.get("character") for item in batch],
        "bboxes": [item.get("bbox") for item in batch],
        "group_ids": [item.get("group_id") for item in batch],
        "shape_indices": [item.get("shape_index") for item in batch],
        "image_paths": [item.get("image_path") for item in batch],
        "json_paths": [item.get("json_path") for item in batch],
        "file_stems": [item.get("file_stem") for item in batch],
        "folders": [item.get("folder") for item in batch],
        "metas": [item.get("meta", {}) for item in batch],
        "raw": batch,
    }


if __name__ == "__main__":
    image_dir = "data/raw/images"
    json_dir = "data/raw/json_files"

    if Path(image_dir).exists() and Path(json_dir).exists():
        dataset = CalligraphyImageDataset(
            image_dir=image_dir,
            json_dir=json_dir,
            image_ext=".jpg",
            image_size=128,
            grayscale=True,
            padding=0.0,
            data_csv="data/raw/data.csv",
            chirography_filter="楷",
        )

        print(f"Loaded single-character crops from LabelMe JSON: {len(dataset)}")

        if len(dataset) > 0:
            s = dataset[0]
            print(
                f"char={s['character']}, "
                f"bbox={s['bbox']}, "
                f"group_id={s['group_id']}, "
                f"shape_index={s['shape_index']}, "
                f"folder={s['folder']}, "
                f"json={s['json_path']}, "
                f"image={s['image_path']}, "
                f"shape={tuple(s['image'].shape)}"
            )
    else:
        print(
            f"image_dir or json_dir not found: "
            f"image_dir={image_dir}, json_dir={json_dir}"
        )
