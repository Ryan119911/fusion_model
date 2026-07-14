# 中文注释：本文件根据 MakeHanzi 中线和真实字符图像构建 B-BSMG 训练用伪配对样本。
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageDraw

from config import load_config, ensure_dirs
from datasets.trajectory_dataset import load_trajectory_csv
from datasets.makehanzi_dataset import MakeHanziDataset
from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from models.dynamic_brush import DynamicBrushModel
from models.geometry import (
    normalize_trajectory_xy,
    dynamic_state_to_bbsmg_input,
)

# 必须与 normalize_trajectory_xy 的 padding 默认值保持一致
NORM_PADDING = 4


# 中文注释：根据点集计算外接矩形。
def bbox_from_points(
    points: List[Tuple[float, float]],
    pad: int = 2,
    canvas_size: int = 128,
) -> Tuple[int, int, int, int]:
    """
    根据归一化后的笔画点，得到该笔画在 128x128 字符画布上的 bbox。
    """
    if len(points) == 0:
        return 0, 0, canvas_size - 1, canvas_size - 1

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x0 = max(int(min(xs)) - pad, 0)
    y0 = max(int(min(ys)) - pad, 0)
    x1 = min(int(max(xs)) + pad, canvas_size - 1)
    y1 = min(int(max(ys)) + pad, canvas_size - 1)

    if x1 <= x0:
        x1 = min(x0 + 1, canvas_size - 1)
    if y1 <= y0:
        y1 = min(y0 + 1, canvas_size - 1)

    return x0, y0, x1, y1


# 中文注释：把折线栅格化为二值图像。
def rasterize_polyline(
    points: List[Tuple[float, float]],
    canvas_size: int = 128,
    width: int = 3,
) -> np.ndarray:
    """
    fallback：当没有真实书法图像 crop 时，直接把轨迹折线栅格化成监督图。
    """
    img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)

    if len(points) >= 2:
        draw.line(points, fill=255, width=width)
    elif len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=255)

    return np.array(img, dtype=np.float32) / 255.0

# 中文注释：根据笔画中线和线宽生成笔画掩码。
def rasterize_stroke_mask(
    points: List[Tuple[float, float]],
    canvas_size: int = 128,
    width: int = 12,
) -> np.ndarray:
    """
    用归一化轨迹生成当前笔画的软/硬 mask。
    mask 尺寸保持为完整 128x128，不裁切、不 resize。
    """
    img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)

    if len(points) >= 2:
        draw.line(points, fill=255, width=width)
    elif len(points) == 1:
        x, y = points[0]
        r = max(width // 2, 1)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    return np.array(img, dtype=np.float32) / 255.0


# 中文注释：从字符画布中裁剪并归一化单笔目标图。
def stroke_target_from_canvas(
    char_canvas: np.ndarray,
    norm_points: List[Tuple[float, float]],
    canvas_size: int = 128,
    mask_width: int = 14,
) -> np.ndarray:
    """
    从真实单字画布中提取当前笔画监督图。

    关键点：
      - 不裁 bbox
      - 不局部 resize
      - 保持完整 128x128 坐标系
      - 用轨迹 mask 只保留当前笔画附近墨迹
    """
    mask = rasterize_stroke_mask(
        norm_points,
        canvas_size=canvas_size,
        width=mask_width,
    )

    target = char_canvas * mask
    return np.clip(target, 0.0, 1.0).astype(np.float32)

# 中文注释：保持宽高比把字符图放入固定画布。
def letterbox_char_to_canvas(
    image_tensor,
    canvas_size: int = 128,
    padding: int = NORM_PADDING,
) -> np.ndarray:
    """
    将真实单字 crop 等比缩放并居中放入 128x128 字符画布。

    这样做的目的是让真实单字 crop 与 normalize_trajectory_xy()
    产生的轨迹坐标处在同一个 128x128 坐标系中。
    """
    arr = image_tensor.detach().cpu().numpy()

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.mean(axis=0)

    hc, wc = arr.shape[:2]

    avail = max(canvas_size - 2 * padding, 1)
    scale = min(avail / max(wc, 1), avail / max(hc, 1))

    new_w = max(1, int(round(wc * scale)))
    new_h = max(1, int(round(hc * scale)))

    crop_img = Image.fromarray(
        (arr * 255.0).clip(0, 255).astype(np.uint8)
    ).resize((new_w, new_h))

    off_x = int(round(padding + (avail - new_w) / 2.0))
    off_y = int(round(padding + (avail - new_h) / 2.0))

    off_x = max(0, min(off_x, canvas_size - new_w))
    off_y = max(0, min(off_y, canvas_size - new_h))

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = (
        np.array(crop_img, dtype=np.float32) / 255.0
    )

    return canvas


# 中文注释：从字符画布中裁剪单笔区域并记录几何信息。
def crop_stroke_from_canvas(
    char_canvas: np.ndarray,
    bbox: Tuple[int, int, int, int],
    image_size: int = 128,
) -> np.ndarray:
    """
    从对齐后的单字画布上，按照该笔画 bbox 裁出局部监督图。
    """
    h, w = char_canvas.shape[:2]
    x0, y0, x1, y1 = bbox

    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))

    if x1 <= x0 or y1 <= y0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    patch = char_canvas[y0:y1 + 1, x0:x1 + 1]

    if patch.size == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    patch_img = Image.fromarray(
        (patch * 255.0).clip(0, 255).astype(np.uint8)
    ).resize((image_size, image_size))

    return np.array(patch_img, dtype=np.float32) / 255.0

# 中文注释：统一目标图前景/背景极性，便于训练。
def normalize_target_polarity(
    target: np.ndarray,
    invert_threshold: float = 0.5,
) -> np.ndarray:
    """
    最终监督图极性统一：
      背景 = 0
      墨迹 = 1

    如果 target 均值过高，通常说明仍是白底黑字或大面积白背景，
    自动反色。
    """
    target = target.astype(np.float32)

    if target.max() > 1.0:
        target = target / 255.0

    target = np.clip(target, 0.0, 1.0)

    if float(target.mean()) > invert_threshold:
        target = 1.0 - target

    return np.clip(target, 0.0, 1.0).astype(np.float32)

# 中文注释：建立 MakeHanzi 字符到中线的索引。
def build_makehanzi_index(dataset: MakeHanziDataset) -> Dict[str, Dict[str, Any]]:
    """
    MakeMeAHanzi 字符索引：
      char -> sample
    """
    return {sample["character"]: sample for sample in dataset.samples}


# 中文注释：根据数据表建立字符到图片路径的索引。
def build_image_index(dataset: CalligraphyImageDataset) -> Dict[str, List[Dict[str, Any]]]:
    """
    懒加载版本：
    只建立 character -> metadata 的索引。
    不在这里打开图片，也不在这里裁图。
    """
    index: Dict[str, List[Dict[str, Any]]] = {}

    for item in dataset.index:
        ch = item.get("character")

        if ch is None:
            continue

        index.setdefault(ch, []).append({
            "item": item,
            "image_path": str(item.get("image_path")),
            "json_path": str(item.get("json_path")),
            "bbox": item.get("bbox"),
            "group_id": item.get("group_id"),
            "shape_index": item.get("shape_index"),
            "folder": item.get("folder"),
            "file_stem": item.get("file_stem"),
        })

    return index

# 中文注释：计算二维折线长度。
def polyline_length(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += float((dx * dx + dy * dy) ** 0.5)
    return total

# 中文注释：兼容不同字段名提取 MakeHanzi 中线。
def _get_makehanzi_medians(makehanzi_sample: Dict[str, Any]) -> List[Any]:
    """
    兼容 MakeHanziDataset 的 sample 结构。
    """
    if makehanzi_sample is None:
        return []

    graphics = makehanzi_sample.get("graphics")
    if graphics is None:
        return []

    return getattr(graphics, "medians", []) or []


# 中文注释：解析命令行参数，准备日志文件并分派到对应子命令。
def main(args):
    cfg = load_config(args.config)

    # 命令行 --chirography 覆盖配置里的 chirography_filter
    if args.chirography is not None:
        cfg.data.chirography_filter = args.chirography

    ensure_dirs(cfg)

    canvas_size = int(cfg.data.canvas_size)
    output_npz = args.output_npz

    print("[INFO] Loading trajectories...")
    traj_samples = load_trajectory_csv(cfg.data.trajectory_csv)
    print(f"[INFO] Loaded trajectory samples: {len(traj_samples)}")

    print("[INFO] Loading MakeMeAHanzi...")
    makehanzi = MakeHanziDataset(
        cfg.data.dictionary_txt,
        cfg.data.graphics_txt,
    )
    makehanzi_index = build_makehanzi_index(makehanzi)
    print(f"[INFO] Loaded MakeMeAHanzi chars: {len(makehanzi_index)}")

    print("[INFO] Loading LabelMe calligraphy images...")
    image_index: Dict[str, List[Dict[str, Any]]] = {}

    if Path(cfg.data.image_dir).exists() and Path(cfg.data.json_dir).exists():
        image_ds = CalligraphyImageDataset(
            image_dir=cfg.data.image_dir,
            json_dir=cfg.data.json_dir,
            image_ext=cfg.data.image_ext,
            image_size=None,      # 保留原始单字 crop 尺寸，后面 letterbox 到 128
            grayscale=True,
            padding=0.0,
            data_csv=getattr(cfg.data, "data_csv", None),
            chirography_filter=getattr(cfg.data, "chirography_filter", None),
        )
        image_index = build_image_index(image_ds)
        print(f"[INFO] Loaded single-character image crops: {len(image_ds)}")
        print(f"[INFO] Image index chars: {len(image_index)}")
    else:
        print(
            f"[WARN] image_dir or json_dir not found: "
            f"image_dir={cfg.data.image_dir}, json_dir={cfg.data.json_dir}"
        )

    brush = DynamicBrushModel()

    inputs: List[List[float]] = []
    targets: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    # 用于同一个字有多个真实 crop 时轮流取样，避免永远取第一个
    image_pick_counter: Dict[str, int] = {}

    print("[INFO] Building pseudo pairs...")

    for sample_idx, traj in enumerate(traj_samples):
        ch = traj.character

        if ch is None or ch == "":
            continue

        strokes = traj.sorted_strokes()
        if len(strokes) == 0:
            continue

        # 轨迹归一化到 128x128 字符画布
        norm_strokes = normalize_trajectory_xy(
            traj,
            canvas_size=canvas_size,
            padding=NORM_PADDING,
        )

        # 动态笔触仿真
        states_by_stroke = brush.simulate_character(
            traj,
            reset_each_stroke=True,
        )

        # MakeMeAHanzi 信息只用于记录和检查
        mh_sample = makehanzi_index.get(ch)
        medians = _get_makehanzi_medians(mh_sample)

        for stroke_order, stroke in enumerate(strokes):
            sid = stroke.stroke_id
            states = states_by_stroke.get(sid, [])

            if len(states) == 0:
                continue

            if stroke_order >= len(norm_strokes):
                continue

            norm_points = norm_strokes[stroke_order]

            if len(norm_points) == 0:
                continue

            # 取该笔画第一个动态状态作为 B-BSMG 条件
            state0 = states[0]

            # 用归一化后的笔画起点覆盖 x0/y0，
            # 否则原始 mm 坐标会和 128x128 图像坐标不一致。
            x0, y0 = norm_points[0]
            x1, y1 = norm_points[-1]
            dx = x1 - x0
            dy = y1 - y0
            length = polyline_length(norm_points)

            bb_input = dynamic_state_to_bbsmg_input(
                state0,
                x0=float(x0),
                y0=float(y0),
            )

            input_vec = [
                float(bb_input.h),
                float(bb_input.alpha),
                float(bb_input.beta),
                float(x0),
                float(y0),
                float(x1),
                float(y1),
                float(dx),
                float(dy),
                float(length),
            ]

            # 优先使用真实书法单字 crop
            used_real_image = False
            source_info: Dict[str, Any] = {}

            if ch in image_index and len(image_index[ch]) > 0:
                candidates = image_index[ch]
                pick_i = image_pick_counter.get(ch, 0) % len(candidates)
                image_pick_counter[ch] = image_pick_counter.get(ch, 0) + 1

                picked = candidates[pick_i]

                # 只有真正用到这个字时，才打开图片并裁单字
                picked_sample = image_ds._build_sample(picked["item"])

                char_canvas = letterbox_char_to_canvas(
                    picked_sample["image"],
                    canvas_size=canvas_size,
                    padding=NORM_PADDING,
                )

                stroke_bbox = bbox_from_points(
                    norm_points,
                    pad=2,
                    canvas_size=canvas_size,
                )

                target = stroke_target_from_canvas(
                    char_canvas=char_canvas,
                    norm_points=norm_points,
                    canvas_size=canvas_size,
                    mask_width=8,
                )

                used_real_image = True
                source_info = {
                    "image_path": picked.get("image_path"),
                    "json_path": picked.get("json_path"),
                    "char_bbox": picked.get("bbox"),
                    "group_id": picked.get("group_id"),
                    "shape_index": picked.get("shape_index"),
                    "folder": picked.get("folder"),
                    "file_stem": picked.get("file_stem"),
                    "stroke_bbox_on_canvas": stroke_bbox,
                }

            else:
                # fallback：无真实图像时，直接用轨迹折线作为监督图
                target = rasterize_polyline(
                    norm_points,
                    canvas_size=canvas_size,
                    width=5,
                )

            # 进入训练集前，再统一一次极性
            target = normalize_target_polarity(target)

            # target_mean = float(target.mean())

            # if target_mean < 1e-4:
            #     continue

            # if target_mean > 0.7:
            #     continue
            
            inputs.append(input_vec)
            targets.append(target)


            metas.append({
                "character": ch,
                "sample_id": traj.meta.get("sample_id"),
                "trajectory_sample_index": sample_idx,
                "stroke_id": sid,
                "stroke_order": stroke_order,
                "num_strokes_in_traj": len(strokes),
                "num_medians_in_makehanzi": len(medians),
                "used_real_image": used_real_image,
                **source_info,
            })

    if len(inputs) == 0:
        raise RuntimeError(
            "No pseudo pairs generated. "
            "请检查：1）trajectories.csv 是否有 character；"
            "2）轨迹字符是否能在 MakeMeAHanzi / JSON 图片标注中匹配；"
            "3）cfg.data.image_dir / cfg.data.json_dir 是否正确。"
        )

    inputs_arr = np.asarray(inputs, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    metas_arr = np.asarray(metas, dtype=object)

    if inputs_arr.shape[1] != cfg.bbsmg.input_dim:
        raise ValueError(
            f"Generated input dimension {inputs_arr.shape[1]} does not match "
            f"config bbsmg.input_dim={cfg.bbsmg.input_dim}."
        )

    out_path = Path(output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        inputs=inputs_arr,
        targets=targets_arr,
        meta=metas_arr,
    )

    print("[DONE] Pseudo pairs saved.")
    print(f"[DONE] output_npz: {out_path}")
    print(f"[DONE] inputs shape : {inputs_arr.shape}")
    print(f"[DONE] targets shape: {targets_arr.shape}")
    print(f"[DONE] meta length   : {len(metas_arr)}")

    real_count = sum(1 for m in metas if m.get("used_real_image"))
    fake_count = len(metas) - real_count

    print(f"[DONE] real image targets : {real_count}")
    print(f"[DONE] rasterized targets : {fake_count}")


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to yaml config",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default="data/processed/bbsmg_train.npz",
        help="Output npz path",
    )
    parser.add_argument(
        "--chirography",
        type=str,
        default=None,
        help="保留参数。当前 LabelMe JSON 数据集中暂未使用书体筛选。",
    )

    args = parser.parse_args()
    main(args)
