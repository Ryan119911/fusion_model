import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable

from torch.utils.data import Dataset

from utils.types import MakeHanziDictionaryRecord, MakeHanziGraphicsRecord


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"MakeMeAHanzi file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_dictionary_records(dictionary_path: str) -> List[MakeHanziDictionaryRecord]:
    rows = _read_jsonl(dictionary_path)
    records: List[MakeHanziDictionaryRecord] = []
    for row in rows:
        records.append(
            MakeHanziDictionaryRecord(
                character=row.get("character", ""),
                pinyin=row.get("pinyin", ""),
                definition=row.get("definition"),
                decomposition=row.get("decomposition"),
                radical=row.get("radical"),
                matches=row.get("matches"),
                etymology=row.get("etymology"),
            )
        )
    return records


def load_graphics_records(graphics_path: str) -> List[MakeHanziGraphicsRecord]:
    rows = _read_jsonl(graphics_path)
    records: List[MakeHanziGraphicsRecord] = []
    for row in rows:
        medians_raw = row.get("medians", [])
        medians: List[List[Tuple[int, int]]] = []
        for stroke_median in medians_raw:
            medians.append([(int(pt[0]), int(pt[1])) for pt in stroke_median])
        records.append(
            MakeHanziGraphicsRecord(
                character=row.get("character", ""),
                strokes=row.get("strokes", []),
                medians=medians,
            )
        )
    return records


def build_dictionary_index(records: List[MakeHanziDictionaryRecord]) -> Dict[str, MakeHanziDictionaryRecord]:
    return {record.character: record for record in records}


def build_graphics_index(records: List[MakeHanziGraphicsRecord]) -> Dict[str, MakeHanziGraphicsRecord]:
    return {record.character: record for record in records}


def join_makehanzi_records(dictionary_records: List[MakeHanziDictionaryRecord], graphics_records: List[MakeHanziGraphicsRecord]) -> List[Dict[str, Any]]:
    dict_index = build_dictionary_index(dictionary_records)
    graphics_index = build_graphics_index(graphics_records)
    characters = sorted(set(dict_index.keys()) | set(graphics_index.keys()))
    joined: List[Dict[str, Any]] = []
    for ch in characters:
        joined.append({
            "character": ch,
            "dictionary": dict_index.get(ch),
            "graphics": graphics_index.get(ch),
        })
    return joined


class MakeHanziDataset(Dataset):
    def __init__(self, dictionary_path: str, graphics_path: str, transform: Optional[Callable[[Dict[str, Any]], Any]] = None, characters: Optional[List[str]] = None):
        self.dictionary_records = load_dictionary_records(dictionary_path)
        self.graphics_records = load_graphics_records(graphics_path)
        self.samples = join_makehanzi_records(self.dictionary_records, self.graphics_records)
        if characters is not None:
            char_set = set(characters)
            self.samples = [sample for sample in self.samples if sample["character"] in char_set]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.transform is not None:
            return self.transform(sample)
        return sample


def extract_character_strokes(sample: Dict[str, Any]) -> List[str]:
    graphics = sample.get("graphics")
    if graphics is None:
        return []
    return graphics.strokes


def extract_character_medians(sample: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
    graphics = sample.get("graphics")
    if graphics is None:
        return []
    return graphics.medians


if __name__ == "__main__":
    dict_path = "data/raw/makemeahanzi/dictionary.txt"
    graphics_path = "data/raw/makemeahanzi/graphics.txt"
    if Path(dict_path).exists() and Path(graphics_path).exists():
        dataset = MakeHanziDataset(dict_path, graphics_path)
        print(f"Loaded MakeMeAHanzi samples: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            graphics = sample.get("graphics")
            num_strokes = len(graphics.strokes) if graphics is not None else 0
            print(f"Character: {sample['character']}, strokes: {num_strokes}")

# 使用说明：该模块用于读取 MakeMeAHanzi 提供的 dictionary.txt 与 graphics.txt 两个 JSON 行文本文件，并按照共同字段 character 自动关联。
# load_dictionary_records() 和 load_graphics_records() 分别解析两类记录；join_makehanzi_records() 会把字典信息与笔画图形信息合并为统一样本；
# MakeHanziDataset 则提供 PyTorch Dataset 接口，便于后续按字读取 SVG path 与 median 骨架。extract_character_strokes() 可直接提取一个字的笔画 SVG path 列表，extract_character_medians() 可直接提取同顺序的笔画中轴点序列。
# 这些信息后续会在 geometry.py 中用于坐标变换、骨架采样、笔画起点估计与伪监督数据构造。
