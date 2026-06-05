"""
数据加载模块 — 图像对读取、标定参数解析、真值加载、序列识别。
"""

import os
import re
import json
import cv2
import numpy as np


def _imread(path):
    """OpenCV imread 的 Unicode 安全替代 (cv2.imread 不支持中文路径)。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data is None or len(data) == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _imread_gray(path):
    """同上，灰度模式。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data is None or len(data) == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


class DataLoader:
    """WoodScape FV 数据加载器。"""

    def __init__(self, data_dir, rgb_dir=None, prev_dir=None, calib_dir=None, gt_dir=None):
        self.data_dir = data_dir
        self.rgb_dir = rgb_dir or os.path.join(data_dir, "rgb_images")
        self.prev_dir = prev_dir or os.path.join(data_dir, "previous_images")
        self.calib_dir = calib_dir or os.path.join(data_dir, "calibration_data")
        self.gt_dir = gt_dir or os.path.join(data_dir, "motion_annotation", "GroudTruth")

        # 扫描所有 ID
        self.ids = self._scan_ids()
        self.ids.sort()

        # 识别连续序列
        self.sequences = self._find_sequences()

        # 预加载标定参数
        self._intrinsic = None

    def _scan_ids(self):
        """扫描 rgb_images 获取所有有效 ID。"""
        ids = []
        for fname in os.listdir(self.rgb_dir):
            m = re.match(r'(\d+)_FV\.png', fname)
            if m:
                ids.append(int(m.group(1)))
        return sorted(ids)

    def _find_sequences(self, min_len=3):
        """识别连续帧序列。"""
        seqs = []
        cur = [self.ids[0]]
        for i in range(1, len(self.ids)):
            if self.ids[i] == self.ids[i - 1] + 1:
                cur.append(self.ids[i])
            else:
                if len(cur) >= min_len:
                    seqs.append(list(cur))
                cur = [self.ids[i]]
        if len(cur) >= min_len:
            seqs.append(list(cur))
        return seqs

    @property
    def intrinsic(self):
        """获取相机内参 (所有帧共享)。"""
        if self._intrinsic is None:
            with open(os.path.join(self.calib_dir, f"{self.ids[0]:05d}_FV.json")) as f:
                self._intrinsic = json.load(f)["intrinsic"]
        return self._intrinsic

    def load_pair(self, img_id):
        """加载一对图像 (当前帧 + 前一帧) 和真值。

        Returns:
            dict: {
                'id': int, 'curr': np.ndarray (H,W,3),
                'prev': np.ndarray (H,W,3), 'gt': np.ndarray (H,W),
                'calib': dict
            }
        """
        fid = f"{img_id:05d}_FV"
        curr = _imread(os.path.join(self.rgb_dir, f"{fid}.png"))
        prev = _imread(os.path.join(self.prev_dir, f"{fid}_prev.png"))
        gt_path = os.path.join(self.gt_dir, f"{fid}.png")

        gt = None
        gt_raw = None
        if os.path.exists(gt_path):
            gt_raw = _imread_gray(gt_path)
            gt = (gt_raw > 0).astype(np.uint8)  # 二值化: 0=static, 1=motion

        with open(os.path.join(self.calib_dir, f"{fid}.json"), encoding='utf-8') as f:
            calib = json.load(f)

        return {
            "id": img_id,
            "curr": curr,
            "prev": prev,
            "gt": gt,
            "gt_raw": gt_raw,
            "calib": calib,
        }

    def iter_pairs(self, sample_ids=None):
        """迭代所有图像对。

        Args:
            sample_ids: 指定 ID 列表, None 表示全部
        """
        ids = sample_ids if sample_ids else self.ids
        for img_id in ids:
            yield self.load_pair(img_id)

    def iter_sequence(self, seq_index=0):
        """迭代连续序列中的所有帧对。

        Args:
            seq_index: 序列索引 (0-based)

        Yields:
            (frame_t, frame_t_plus_1, gt_t_plus_1) 用于跟踪实验
        """
        if seq_index >= len(self.sequences):
            raise IndexError(f"Sequence {seq_index} out of range ({len(self.sequences)} seqs)")

        seq_ids = self.sequences[seq_index]
        pairs = [self.load_pair(sid) for sid in seq_ids]

        # 返回连续帧: (frame[i], frame[i+1])
        for i in range(len(pairs) - 1):
            yield {
                "curr": pairs[i + 1]["curr"],
                "prev": pairs[i + 1]["prev"],  # 前一对的 curr = 这一对的 prev
                "next": pairs[i + 1]["curr"],
                "prev_frame": pairs[i]["curr"],  # 真实前一帧
                "curr_gt": pairs[i + 1]["gt"],
                "prev_gt": pairs[i]["gt"],
                "id_curr": pairs[i + 1]["id"],
                "id_prev": pairs[i]["id"],
            }

    def get_sequence_ids(self, seq_index=0):
        """获取指定序列的 ID 列表。"""
        if seq_index < len(self.sequences):
            return self.sequences[seq_index]
        return []

    def print_summary(self):
        """打印数据集摘要信息。"""
        print("=" * 60)
        print("数据集摘要")
        print("=" * 60)
        print(f"  总样本数:     {len(self.ids)}")
        print(f"  连续序列数:   {len(self.sequences)} (>=3帧)")
        print(f"  图像尺寸:     {int(self.intrinsic['width'])} x {int(self.intrinsic['height'])}")
        print(f"  标定模型:     {self.intrinsic['model']} (order={self.intrinsic['poly_order']})")
        print(f"  参数 k1-k4:   {self.intrinsic['k1']:.3f}, {self.intrinsic['k2']:.3f}, {self.intrinsic['k3']:.3f}, {self.intrinsic['k4']:.3f}")
        print(f"  主点偏移:     cx={self.intrinsic['cx_offset']:.3f}, cy={self.intrinsic['cy_offset']:.3f}")
        print("-" * 60)
        for i, s in enumerate(self.sequences):
            print(f"  Seq-{chr(65+i)}: {len(s)} 帧  [{s[0]:05d} – {s[-1]:05d}]")
        print("=" * 60)
