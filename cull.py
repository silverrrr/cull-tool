#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粗筛合焦照片 —— 人像/活动向的对焦预筛

思路:
  大光圈人像的清晰度必须看「眼睛」, 不能看整幅画面。
  1. rawpy(libraw) 真正解码 RAW, 拿到原始像素
  2. OpenCV YuNet(轻量 CNN 人脸检测) 找脸, 并给出眼睛等 5 个关键点
  3. 在眼睛位置的原始像素上算 Tenengrad 梯度能量, 再除以该区域的对比度做
     归一化 —— 合焦的眼睛边缘锐利、得分高, 跑焦的边缘被抹平、得分低;
     除以对比度是为了抵消阴影/欠曝的影响
  4. 把「脸的位置 + 脸的大小 + 焦距」相近的帧归为一组 (同一姿势的连拍),
     只在组内比较锐度 —— 不同距离/焦段的照片锐度本来就没有可比性
  5. 输出: CSV + 可视化 HTML(带人脸框和眼睛标记) + 组内最佳清单 + 疑似模糊清单

用法:
  run.bat <照片目录>                    # 半分辨率解码, 快
  run.bat <照片目录> --full              # 全分辨率解码, 更准
  run.bat <照片目录> --ratio 0.6         # 疑似模糊灵敏度 (默认 0.55)
  run.bat <照片目录> --sheet             # 额外导出一张检测结果总览图
"""

import argparse
import base64
import csv
import datetime
import html
import io
import itertools
import json
import math
import os
import pathlib
import shutil
import struct
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image, ImageDraw

RAW_EXTS = (".arw", ".sr2", ".sr3", ".nef", ".nrw", ".cr2", ".cr3", ".dng", ".orf", ".rw2", ".raf", ".pef")
# Pillow 能解的都收进来。注意 decode() 必须同步认这些后缀, 否则它们会掉进
# rawpy 分支, 然后报"找不到内嵌预览图" (那是给 RAW 用的兜底, 对普通图片没意义)。
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

DET_W = 1600
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_NAME = "face_detection_yunet_2023mar.onnx"

EXIF_TAGS = {
    0x829A: ("exposure_time", "rational"),
    0x829D: ("f_number", "rational"),
    0x8827: ("iso", "short"),
    0x9003: ("datetime_original", "ascii"),
    0x920A: ("focal_length", "rational"),
    0xA434: ("lens_model", "ascii"),
}


# ==========================================================================
# 模型下载 / 存放 / 中文路径
#
# 模型统一放在 exe 同级的 _internal\models\ (打包后 sys._MEIPASS 指向 _internal;
# 源码运行则是工具目录下的 models\)。首次运行按需下载, 之后纯离线。
# 可用 cull_config.json 的 model_dir / CLI --model 覆盖。
# ==========================================================================
LMK_URL = ("https://huggingface.co/public-data/insightface/resolve/main/"
           "models/buffalo_l/2d106det.onnx")
MUSIQ_ONNX_URL = ("https://huggingface.co/86Cao/IQA-ONNX-Models/resolve/main/musiq_model.onnx")
MUSIQ_DATA_URL = ("https://huggingface.co/86Cao/IQA-ONNX-Models/resolve/main/"
                  "musiq_model.onnx.data")
NIMA_URL = ("https://huggingface.co/cromsc/nima-mobilenet-aesthetic/resolve/main/"
            "nima_mobilenet_aesthetic.onnx")

# name -> [(文件名, url, 最小字节)]。musiq 用外部数据格式(.onnx + .onnx.data), 两个都要下。
ASSETS = {
    "yunet": [(YUNET_NAME, YUNET_URL, 100_000)],
    "landmark": [("2d106det.onnx", LMK_URL, 3_000_000)],
    "musiq": [("musiq_model.onnx", MUSIQ_ONNX_URL, 500_000),
              ("musiq_model.onnx.data", MUSIQ_DATA_URL, 50_000_000)],
    "nima": [("nima_mobilenet_aesthetic.onnx", NIMA_URL, 10_000_000)],
}

# 默认配置 (可被 cull_config.json 覆盖)
DEFAULT_CFG = {
    "model_dir": None,                       # 覆盖模型存放目录
    "enable": {"landmark": True, "musiq": True, "nima": True},
    "weights": {"sharp": 0.30, "tech": 0.20, "pose": 0.20, "expr": 0.15, "aes": 0.15},
}
_CFG = json.loads(json.dumps(DEFAULT_CFG))   # 深拷贝


def load_config():
    """读工具目录/exe 同级的 cull_config.json, 覆盖到 _CFG 上。没有就用默认。"""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(base, os.pardir, "cull_config.json"),
                 os.path.join(base, "cull_config.json")):
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                for k, v in cfg.items():
                    if isinstance(v, dict) and isinstance(_CFG.get(k), dict):
                        _CFG[k].update(v)
                    else:
                        _CFG[k] = v
                break
            except Exception:                                # noqa: BLE001
                pass
    return _CFG


def model_dir():
    """模型目录: 打包后 = exe 同级 _internal\\models; 源码 = 工具目录\\models。"""
    d = _CFG.get("model_dir")
    if not d:
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(base, "models")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def _ascii_dir():
    """OpenCV 在 Windows 下打不开含中文路径的文件 —— 镜像到这里再加载。"""
    for cand in (os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "cmdc_cull"),
                 os.path.join(tempfile.gettempdir(), "cmdc_cull")):
        try:
            cand.encode("ascii")
            os.makedirs(cand, exist_ok=True)
            return cand
        except (UnicodeEncodeError, OSError):
            continue
    return None


def ensure_asset(key):
    """按需下载某组模型文件, 返回主模型路径。失败抛 RuntimeError。"""
    d = model_dir()
    first = None
    for fname, url, min_size in ASSETS[key]:
        dst = os.path.join(d, fname)
        if not os.path.isfile(dst) or os.path.getsize(dst) < min_size:
            print(f"首次运行: 下载模型 {fname} ...")
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as exc:                         # noqa: BLE001
                raise RuntimeError(
                    f"模型下载失败 ({fname}): {exc}\n"
                    f"可手动下载后放到 {os.path.join(d, fname)}, "
                    f"或用 cull_config.json 的 model_dir 指定目录") from exc
        if first is None:
            first = dst
    return first


def _opencv_loadable(path):
    """OpenCV 在 Windows 下打不开含中文的路径 —— 非 ASCII 时镜像到英文临时目录。"""
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    d = _ascii_dir()
    if not d:
        return path
    mirror = os.path.join(d, os.path.basename(path))
    try:
        if not os.path.isfile(mirror) or os.path.getsize(mirror) != os.path.getsize(path):
            shutil.copy2(path, mirror)
    except OSError:
        return path
    return mirror


def ensure_model(path=None):
    """YuNet 检测模型。--model 指定就用它, 否则按需下载到 model_dir。"""
    if path and os.path.isfile(path):
        return _opencv_loadable(path)
    return _opencv_loadable(ensure_asset("yunet"))


def make_detector(model_path, score_thr=0.6):
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                     score_threshold=score_thr,
                                     nms_threshold=0.3, top_k=5000)


# ==========================================================================
# 关键点 / 头部姿态 / 睁眼 / 画质 / 美感   (onnxruntime)
# ==========================================================================
def make_ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1     # 和 OpenCV 同理: 多进程时别开满线程抢核
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def detect_landmarks(bgr, box, sess):
    """insightface 2d106det: 192x192 裁剪 (scale=192/(max(w,h)*1.5)), (x-127.5)/128。

    返回 106x2 的关键点 (原图坐标)。
    """
    x0, y0, bw, bh = box
    cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
    m = max(bw, bh, 1.0)
    s = 192.0 / (m * 1.5)
    M = np.array([[s, 0, -cx * s + 96.0], [0, s, -cy * s + 96.0]], np.float32)
    aimg = cv2.warpAffine(bgr, M, (192, 192), borderValue=0)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / 128.0, (192, 192),
                                 (127.5, 127.5, 127.5), swapRB=True)
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].reshape(-1, 2)
    pred = (pred + 1.0) * 96.0
    IM = cv2.invertAffineTransform(M)                    # 2x3, 把裁剪坐标映射回原图
    return pred @ IM[:, :2].T + IM[:, 2]


def eye_metrics(pts, lms, k=10):
    """算 (睁眼程度 EAR, 偏航估计 yaw°)。

    不用固定的 106 索引表 —— 取 106 点里离 YuNet 眼点最近的 k 个当眼周轮廓:
      - EAR = 眼高 / 眼宽, 两眼取小 (眨眼会让它掉下来)
      - yaw: 侧脸时远侧眼被压缩, 两眼投影宽度之比 ≈ cos(yaw) → yaw = acos(比值)。
        比 5 点 solvePnP 稳得多 (后者在近平面点上会给出 -170° 这种离谱值)。
    """
    ears, widths = [], []
    for eye in (lms[0], lms[1]):
        d = np.linalg.norm(pts - eye, axis=1)
        sel = pts[np.argsort(d)[:k]]
        w = float(sel[:, 0].max() - sel[:, 0].min()) + 1e-6
        widths.append(w)
        bins = np.linspace(sel[:, 0].min(), sel[:, 0].max(), 5)
        ups, lows = [], []
        for i in range(4):
            m = (sel[:, 0] >= bins[i]) & (sel[:, 0] <= bins[i + 1])
            if m.any():
                ups.append(sel[m][:, 1].min())
                lows.append(sel[m][:, 1].max())
        if ups:
            ears.append(float(np.mean(np.array(lows) - np.array(ups))) / w)
    ear = min(ears) if ears else None
    wr = min(widths) / max(widths) if max(widths) > 0 else 1.0
    # 实测: 正面 wr≈0.90, 3/4≈0.55, 侧脸≈0.40。线性映射到 0~75°, 不做 acos
    # (acos 会把正面的 0.90 也算成 26°, 系统性抬高)。
    yaw = 75.0 * max(0.0, min(1.0, (0.90 - wr) / 0.55))
    return ear, yaw


def tech_quality(bgr, sess):
    """MUSIQ 技术画质 (0~100)。输入 224x224, 归一化 (x-0.5)/0.5。"""
    x = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[np.newaxis]
    out = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]
    return float(np.asarray(out).ravel()[0])


def aesthetic_score(bgr, sess):
    """NIMA 美感 (1~10): 输出 10 个 bin 的概率, 取期望值。"""
    x = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: x[np.newaxis].astype(np.float32)})[0][0]
    return float((np.arange(1, 11) * out).sum())


load_config()


# ==========================================================================
# 解码
# ==========================================================================
def decode(path, full=False):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXTS:
        return cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR), False
    try:
        import rawpy
        with rawpy.imread(path) as raw:
            bgr = raw.postprocess(half_size=not full, use_camera_wb=True, output_bps=8)
        return cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR), False
    except Exception:                                        # noqa: BLE001
        blob, dims, _ = extract_preview(path)
        return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR), True


# ---- 内嵌预览兜底 --------------------------------------------------------
def _read_jpeg_at(data, start):
    if data[start:start + 3] != b"\xff\xd8\xff":
        return None
    j, dims, end = start + 2, None, len(data)
    while j + 1 < end:
        if data[j] != 0xFF:
            return None
        m = data[j + 1]
        if m == 0xFF:
            j += 1
            continue
        if m == 0xD8 or m == 0x01 or 0xD0 <= m <= 0xD7:
            j += 2
            continue
        if m == 0xD9 or j + 4 > end:
            return None
        seglen = struct.unpack(">H", data[j + 2:j + 4])[0]
        if m == 0xDA:
            eoi = data.find(b"\xff\xd9", j + 2 + seglen)
            return (eoi + 2 - start, dims) if eoi != -1 else None
        if seglen < 2:
            return None
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[j + 5:j + 9])
            dims = (w, h)
        j += 2 + seglen
    return None


def extract_preview(path):
    with open(path, "rb") as fh:
        data = fh.read()
    best, pos = None, data.find(b"\xff\xd8\xff")
    while pos != -1:
        info = _read_jpeg_at(data, pos)
        if info and info[1]:
            area = info[1][0] * info[1][1]
            if best is None or area > best[0]:
                best = (area, pos, info[0], info[1])
        pos = data.find(b"\xff\xd8\xff", pos + 1)
    if best is None:
        raise ValueError("找不到内嵌预览图")
    _, off, size, dims = best
    del data
    with open(path, "rb") as fh:
        fh.seek(off)
        blob = fh.read(size)
    Image.open(io.BytesIO(blob)).verify()
    return blob, dims, dims[0] * dims[1]


def list_photos(folder, exts=None):
    """列出要分析的文件, 返回 (要分析的名字列表, 全部候选数)。

    已经有同名 RAW 的 jpg 导出版会被跳过, 免得同一张算两遍。
    """
    wanted = tuple(e.lower() for e in (exts or (RAW_EXTS + IMG_EXTS)))
    all_names = [n for n in sorted(os.listdir(folder))
                 if os.path.isfile(os.path.join(folder, n))
                 and os.path.splitext(n)[1].lower() in wanted]
    raw_stems = {os.path.splitext(n)[0] for n in all_names
                 if os.path.splitext(n)[1].lower() in RAW_EXTS}
    names = [n for n in all_names
             if os.path.splitext(n)[1].lower() in RAW_EXTS
             or os.path.splitext(n)[0] not in raw_stems]
    return names, len(all_names)


def make_big_jpeg(path, side=1600, full=False):
    """按需解码出大图 JPEG 的 bytes。GUI 点开看大图时用, 不落盘。"""
    bgr, _ = decode(path, full=full)
    m = max(bgr.shape[:2])
    if m > side:
        bgr = cv2.resize(bgr, None, fx=side / m, fy=side / m, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


# 预览 jpg 的文件名后缀。故意加 .preview, 这样 agent 一眼能看出这是工具生成的,
# 也不会跟 Lightroom 导出的同名 .jpg 撞名 (那种是修过图、调过色的, 含义不同)。
PREVIEW_SUFFIX = ".preview.jpg"


def make_preview_jpeg(path, out_path, side=1600):
    """给一张照片写一个预览 jpg, 供人/agent 快速查看。

    **必须重新解码, 不能抽 RAW 里的内嵌预览图** —— 实测 A7R III 的内嵌预览是
    1616x1080 但**转了 90 度**, 而且没有 EXIF Orientation 标记, 任何程序都没法
    自动转正。拿它给 agent 看会给出完全错误的判断 (比如把人像当侧躺着)。

    画检测框之类的都不要: 这张图是给人/AI 看画面内容的, 不是给检测结果看的。
    """
    raw = make_big_jpeg(path, side=side)
    if not raw:
        raise RuntimeError(f"生成预览失败: {os.path.basename(path)}")
    with open(out_path, "wb") as fh:
        fh.write(raw)
    return len(raw)


def preview_name(photo_name):
    """JC_05478.ARW -> JC_05478.preview.jpg"""
    return os.path.splitext(photo_name)[0] + PREVIEW_SUFFIX


# ==========================================================================
# EXIF
# ==========================================================================
def _read_ifd(data, offset, endian, wanted):
    out = {}
    if offset <= 0 or offset + 2 > len(data):
        return out
    count = struct.unpack(endian + "H", data[offset:offset + 2])[0]
    if count > 512:
        return out
    for i in range(count):
        e = offset + 2 + i * 12
        if e + 12 > len(data):
            break
        tag, _t, cnt = struct.unpack(endian + "HHI", data[e:e + 8])
        if tag not in wanted:
            continue
        name, kind = wanted[tag]
        raw = data[e + 8:e + 12]
        if kind == "ascii":
            voff = struct.unpack(endian + "I", raw)[0]
            chunk = raw[:cnt] if cnt <= 4 else data[voff:voff + cnt]
            out[name] = chunk.split(b"\x00")[0].decode("utf-8", "replace").strip()
        elif kind == "short":
            out[name] = struct.unpack(endian + "H", raw[:2])[0]
        elif kind == "long":
            out[name] = struct.unpack(endian + "I", raw)[0]
        elif kind == "rational":
            voff = struct.unpack(endian + "I", raw)[0]
            if voff + 8 <= len(data):
                out[name] = struct.unpack(endian + "II", data[voff:voff + 8])
    return out


def read_exif(path):
    with open(path, "rb") as fh:
        head = fh.read(256 * 1024)
    endian = "<" if head[:4] == b"II*\x00" else (">" if head[:4] == b"MM\x00*" else None)
    if endian is None:
        return {}
    ifd0 = struct.unpack(endian + "I", head[4:8])[0]
    tags = {}
    ptr = _read_ifd(head, ifd0, endian, {0x8769: ("exif_offset", "long")})
    if "exif_offset" in ptr:
        tags.update(_read_ifd(head, ptr["exif_offset"], endian, EXIF_TAGS))
    return tags


def fmt_exposure(et):
    if not et or not et[1]:
        return ""
    v = et[0] / et[1]
    return f"{v:g}s" if v >= 1 else f"1/{round(1 / v)}"


def fmt_rat(r, d=2):
    return round(r[0] / r[1], d) if r and r[1] else ""


# ==========================================================================
# Sony 对焦设置 (maker note 0x201b/0x201c/0x201d, 这三个标签未加密可直接读)
#
# 重要限制: AFAreaMode = Wide 时由相机自行选点, 而 ILCE 机型**不记录**它实际选了
# 哪个点 (那段数据在加密的 0x94xx 块里, 连 ExifTool 都没解开)。所以只有当摄影师
# 用了 Flexible Spot / Center 时, 才拿得到对焦点坐标。
# ==========================================================================
SONY_FOCUS_MODE = {0: "Manual", 2: "AF-S", 3: "AF-C", 4: "AF-A", 6: "DMF", 7: "AF-D"}
SONY_AF_AREA = {0: "Wide", 1: "Zone", 2: "Center", 3: "FlexibleSpot", 4: "ExpandedFlexibleSpot"}
AF_CANVAS = (640.0, 480.0)          # Sony 记录对焦点坐标用的画布


def read_sony_af(path):
    try:
        with open(path, "rb") as fh:
            data = fh.read(160 * 1024)
    except OSError:
        return {}
    if data[:4] == b"II*\x00":
        e = "<"
    elif data[:4] == b"MM\x00*":
        e = ">"
    else:
        return {}

    def ifd(off):
        res = {}
        if off <= 0 or off + 2 > len(data):
            return res
        n = struct.unpack(e + "H", data[off:off + 2])[0]
        if n > 512:
            return res
        for i in range(n):
            p = off + 2 + i * 12
            if p + 12 > len(data):
                break
            tag, typ, cnt = struct.unpack(e + "HHI", data[p:p + 8])
            res[tag] = (typ, cnt, data[p + 8:p + 12])
        return res

    ifd0 = ifd(struct.unpack(e + "I", data[4:8])[0])
    if 0x8769 not in ifd0:
        return {}
    exif = ifd(struct.unpack(e + "I", ifd0[0x8769][2])[0])
    if 0x927C not in exif:
        return {}
    mn = ifd(struct.unpack(e + "I", exif[0x927C][2])[0])

    out = {}
    if 0x201B in mn:                                    # int8u
        v = mn[0x201B][2][0]
        out["focus_mode"] = SONY_FOCUS_MODE.get(v, f"0x{v:02x}")
    if 0x201C in mn:                                    # int8u
        v = mn[0x201C][2][0]
        out["af_area"] = SONY_AF_AREA.get(v, f"0x{v:02x}")
    if 0x201D in mn:                                    # int16u[2], 内联 4 字节
        ax, ay = struct.unpack(e + "HH", mn[0x201D][2])
        if ax or ay:
            out["af_x"], out["af_y"] = ax, ay
            out["af_point"] = (ax / AF_CANVAS[0], ay / AF_CANVAS[1])
    return out


# ==========================================================================
# 多进程: 每个工作进程各持一个 detector (模型只有 230KB, 加载很快)
# ==========================================================================
_W = {}


class HardStopPool:
    """能真正打断的 ProcessPoolExecutor。

    为什么不能直接用 ProcessPoolExecutor:
      shutdown(wait=False, cancel_futures=True) 只是不再派发**还没开始**的任务,
      已经跑起来的那几张照样解码完。而 cancel_futures=True 会把这些在途的 future
      直接标成 CANCELLED, as_completed() 收到的是个还没做完的 future, 调 result()
      立刻抛 CancelledError —— 于是调用方以为"已停", 实际 worker 进程还占着几百 MB
      内存继续读盘。这正是点了取消之后后台还在解码的原因。

    这里改成: 取消时先 terminate 掉所有 worker (正在解码的那几张当场丢掉),
    再 join 回收。只丢结果, 不动源文件。
    """

    def __init__(self, max_workers, initializer=None, initargs=()):
        self._ex = ProcessPoolExecutor(max_workers=max_workers,
                                       initializer=initializer,
                                       initargs=initargs)
        self._closed = False

    def submit(self, fn, *a):
        return self._ex.submit(fn, *a)

    def as_completed(self, futs):
        return as_completed(futs)

    def stop(self):
        """terminate 所有 worker 并回收。幂等, 异常也不怕。"""
        if self._closed:
            return
        self._closed = True
        # _processes 是 {pid: Process}, 要取 values(); 直接遍历拿到的是 pid(int)
        try:
            procs = list((self._ex._processes or {}).values())
        except AttributeError:
            procs = []
        for p in procs:
            try:
                p.terminate()
            except Exception:                            # noqa: BLE001
                pass
        for p in procs:
            try:
                p.join(timeout=5)
            except Exception:                            # noqa: BLE001
                pass
        for p in procs:
            if p.is_alive():
                try:
                    p.kill()
                    p.join(timeout=5)
                except Exception:                        # noqa: BLE001
                    pass
        try:
            self._ex.shutdown(wait=False, cancel_futures=True)
        except Exception:                                # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self._closed:
            self.stop()
        return False


def worker_init(model_path, full, preview_dir=None, preview_side=2048, cv_threads=1):
    # 多进程时要把 OpenCV 自己的线程数压下来: 默认它会开满所有核,
    # 4 个进程 x 20 线程 = 80 线程抢 20 个核, 反而把整体拖慢。
    if cv_threads and cv_threads > 0:
        cv2.setNumThreads(cv_threads)
    _W["det"] = make_detector(model_path)
    _W["det_model"] = model_path      # reselect_subject 要重新建一个检测器
    _W["full"] = full
    _W["preview_dir"] = preview_dir
    _W["preview_side"] = preview_side
    # 关键点 / 画质 / 美感 三组 onnx 会话。缺模型不影响锐度主流程, 只是该项跳过。
    en = _CFG.get("enable", {})
    for key, sk in (("landmark", "lmk"), ("musiq", "musiq"), ("nima", "nima")):
        _W[sk] = None
        if en.get(key, True):
            try:
                _W[sk] = make_ort_session(ensure_asset(key))
            except Exception as exc:                         # noqa: BLE001
                print(f"  [!] {key} 模型不可用, 该项跳过 ({exc})")


def process_one(name, folder):
    path = os.path.join(folder, name)
    rec = {"file": name, "_error": ""}
    try:
        rec["orig_uri"] = pathlib.Path(path).as_uri()
    except Exception:                                       # noqa: BLE001
        rec["orig_uri"] = ""
    af = {}
    try:
        af = read_sony_af(path)
    except Exception:                                       # noqa: BLE001
        pass
    pd = _W.get("preview_dir")
    pv = os.path.join(pd, os.path.splitext(name)[0] + ".jpg") if pd else None
    try:
        rec.update(analyze(path, _W["det"], full=_W["full"], af_point=af.get("af_point"),
                           preview_path=pv, preview_side=_W.get("preview_side", 2048)))
    except Exception as exc:                                # noqa: BLE001
        rec.update({"_error": str(exc), "subject": "?", "compare_value": 0.0,
                    "eye_sharp": 0.0, "face_sharp": 0.0, "sharp_global": 0.0,
                    "faces": 0, "_thumb": "", "preview_rel": "", "_face_box": None,
                    "_eye_boxes": [], "sig": None, "_face_boxes": [],
                    "face_w": 0, "face_cx": 0, "face_cy": 0, "face_area_pct": 0,
                    "face_conf": 0, "brightness": 0, "clipped": 0, "decoded": "", "_ms": 0,
                    "yaw": None, "pitch": None, "roll": None, "ear": None,
                    "mos_tech": None, "aes": None})
    try:
        rec.update(read_exif(path))
    except Exception:                                       # noqa: BLE001
        pass
    rec.update(af)
    return rec


def print_progress(i, total, rec):
    print(f'[{i:>3}/{total}] {rec["file"]:<15} 眼锐度={rec["eye_sharp"]:>8}  脸={rec["faces"]}  '
          f'{fmt_exposure(rec.get("exposure_time")):>7}  f/{fmt_rat(rec.get("f_number"),1):<4} '
          f'{fmt_rat(rec.get("focal_length"),0)}mm  {rec.get("_ms",0)}ms'
          + (f'  AF={rec["af_area"]}' if rec.get("af_area") else "")
          + (f'  !! {rec["_error"]}' if rec.get("_error") else ""))


# ==========================================================================
# 锐度
# ==========================================================================
def tenengrad_map(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def patch_sharpness(gray, x, y, r):
    """以 (x,y) 为中心、半径 r 的方形区域锐度。返回 (tenengrad均值, 对比度)"""
    H, W = gray.shape
    x0, y0 = max(0, int(x - r)), max(0, int(y - r))
    x1, y1 = min(W, int(x + r)), min(H, int(y + r))
    crop = gray[y0:y1, x0:x1]
    if crop.shape[0] < 12 or crop.shape[1] < 12:
        return 0.0, 0.0
    c = cv2.GaussianBlur(crop, (0, 0), 1.0)
    return float(tenengrad_map(c).mean()), float(crop.std())


def face_signature(gray, box, size=64):
    """人脸区域归一化灰度指纹, 用于判断两帧是不是同一个姿势"""
    x, y, w, h = [int(v) for v in box]
    H, W = gray.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    crop = gray[y0:y1, x0:x1]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return None
    s = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    s = (s - s.mean()) / (s.std() + 1e-6)
    return s.ravel()


# ==========================================================================
# 单张分析
# ==========================================================================
PATCH_OUT = 96          # 归一化后的眼框边长 (px)


def _norm_patch(gray, cx, cy, w, h, out):
    """以 (cx,cy) 为中心取 w×h 的区域 (越界用边缘复制补齐), 重采样到 out×out。"""
    H, W = gray.shape
    x0, y0 = int(round(cx - w / 2.0)), int(round(cy - h / 2.0))
    x1, y1 = int(round(cx + w / 2.0)), int(round(cy + h / 2.0))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - W), max(0, y1 - H)
    crop = gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        return None
    if px0 or py0 or px1 or py1:
        crop = cv2.copyMakeBorder(crop, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if max(crop.shape[:2]) > out else cv2.INTER_LINEAR
    return cv2.resize(crop, (out, out), interpolation=interp)


def _ten_patch(patch):
    """固定尺寸 patch 上的 (Tenengrad 均值, 对比度 std)。"""
    c = cv2.GaussianBlur(patch, (0, 0), 1.0)
    return float(tenengrad_map(c).mean()), float(patch.std())


def measure_face(gray, fx, fy, fw, fh, lms, yaw=0.0):
    """量一张脸的眼部锐度。返回 (eye_sharp, eye_boxes)。

    这是选脸和"手动换主体"共用的唯一口径 —— 两者必须算得一模一样,
    否则用户手动换了主体, 分数却对不上。

    锐度 = Tenengrad 均值 ÷ 对比度, 且做了三重归一化:
      1. **尺度**: 眼框尺寸按瞳距取 (0.70×瞳距), 再重采样到固定 PATCH_OUT px,
         消除"框景大小"导致的像素数差异 (远景小脸 vs 近景大脸)
      2. **朝向**: 按 1/cos(yaw) 横向多取样, 抵消侧脸时远侧眼被压缩 —— 修掉
         "3/4 侧脸明明合焦却被评低分"
      3. **对比度**: ÷ crop.std(), 抵消阴影/欠曝 (v3)
    """
    iod = float(np.hypot(*(lms[0] - lms[1])))
    if not (iod > 6):
        iod = max(8.0, fw * 0.5)
    side = max(12.0, 0.70 * iod)                     # 眼框物理尺寸 ∝ 瞳距
    cyaw = math.cos(math.radians(min(50.0, abs(yaw))))
    sx = side / max(0.65, cyaw)                      # 横向多取样, 抵消侧脸的压缩
    eyes = []
    for ex, ey in (lms[0], lms[1]):
        p = _norm_patch(gray, ex, ey, sx, side, PATCH_OUT)
        if p is not None:
            t, c = _ten_patch(p)
            if t > 0:
                eyes.append(t / (c + 1.0))
    eye = float(np.mean(eyes)) if eyes else 0.0
    er = side / 2.0
    boxes = [[int(ex - er), int(ey - er), int(side), int(side)]
             for ex, ey in (lms[0], lms[1])]
    return eye, boxes


def pick_subject(gray, faces, inv, W, H, bgr=None, lmk=None):
    """遍历所有脸: 关键点 → 姿态/睁眼 → 眼部锐度(尺度+朝向归一化)。

    返回 (best, all_boxes, all_metrics)。best 多带 yaw/pitch/roll/ear/idx。
    bgr+lmk 给了就顺便算姿态和睁眼; 没有(或模型缺失)则 yaw=0、ear=None。
    """
    best = None
    all_boxes = []
    metrics = []
    for idx, f in enumerate(faces):
        fx, fy, fw, fh = [float(v) * inv for v in f[:4]]
        lms = (f[4:14].reshape(5, 2) * inv)
        conf = float(f[-1])
        yaw = pitch = roll = ear = None
        if bgr is not None and lmk is not None:
            try:
                lpts = detect_landmarks(bgr, (fx, fy, fw, fh), lmk)
                ear, yaw = eye_metrics(lpts, lms)
            except Exception:                                # noqa: BLE001
                pass
        # numpy 标量 (np.float32) 不能 json 序列化, 而 face_cx/face_cy/
        # face_area_pct 都由它们算出来 —— 不先转成 python float 的话
        # write_cache() 会对每一张都静默失败。这里统一转掉。
        eye, eboxes = measure_face(gray, fx, fy, fw, fh, lms, yaw or 0.0)
        all_boxes.append([int(fx), int(fy), int(fw), int(fh)])
        metrics.append({"idx": idx, "eye": round(eye, 1), "w": int(fw),
                        "cx": round((fx + fw / 2) / W, 4),
                        "cy": round((fy + fh / 2) / H, 4),
                        "conf": round(conf, 2),
                        "yaw": round(yaw, 1) if yaw is not None else None,
                        "ear": round(ear, 3) if ear is not None else None})
        cand = {"eye": eye, "fx": fx, "fy": fy, "fw": fw, "fh": fh,
                "lms": lms, "conf": conf, "er": max(10.0, fw * 0.24),
                "eye_boxes": eboxes, "idx": idx,
                "yaw": yaw, "pitch": pitch, "roll": roll, "ear": ear}
        if best is None or cand["eye"] > best["eye"]:
            best = cand
    return best, all_boxes, metrics


# ==========================================================================
# 换主体: 用户在界面上点了"第 N 张脸才是主角"
# ==========================================================================
def reselect_subject(row, name, folder, idx, full=False, af_point=None):
    """把 row 的主体换成第 idx 张脸, 原地更新并返回它。

    为什么要重新解码: 换了主体之后 `sig`(人脸指纹) 和 `face_sharp` 都得跟着
    变, 而这俩都需要原图像素 —— 缓存里只存了缩略图, 算不出来。
    所以"换主体"这一次是按需重解码单张 (~0.3s), 换完立刻写回缓存。
    """
    path = os.path.join(folder, name)
    bgr, from_preview = decode(path, full=full)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    sc = DET_W / max(H, W)
    small = (cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
             if sc < 1 else gray)
    det = make_detector(_W["det_model"]) if _W.get("det_model") else None
    if det is None:
        det = make_detector(ensure_model(None))
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR))
    faces = [] if faces is None else faces
    if idx < 0 or idx >= len(faces):
        raise ValueError("这张照片没有那么多张脸 (检出 %d 张)" % len(faces))
    inv = 1.0 / sc
    f = faces[idx]
    fx, fy, fw, fh = [float(v) * inv for v in f[:4]]
    lms = (f[4:14].reshape(5, 2) * inv)
    conf = float(f[-1])
    eye, eboxes = measure_face(gray, fx, fy, fw, fh, lms)
    face_ten, _ = patch_sharpness(gray, fx + fw / 2, fy + fh / 2, max(fw, fh) * 0.55)
    row.update({
        "eye_sharp": round(eye, 1),
        "face_sharp": round(face_ten, 1),
        "face_conf": round(conf, 2),
        "face_w": int(fw),
        "face_cx": round((fx + fw / 2) / W, 4),
        "face_cy": round((fy + fh / 2) / H, 4),
        "face_area_pct": round(fw * fh / (W * H) * 100, 2),
        "sig": face_signature(gray, (fx, fy, fw, fh)),
        "_face_box": [int(fx), int(fy), int(fw), int(fh)],
        "_eye_boxes": eboxes,
        "subject_idx": idx,
        "compare_value": round(eye, 1),
    })
    row["_thumb"] = make_thumb(bgr, row["_face_box"], eboxes, af_point,
                               all_face_boxes=row.get("_face_boxes"))
    return row


def analyze(path, detector, full=False, af_point=None, preview_path=None, preview_side=2048):
    rec = {}
    t0 = time.time()
    bgr, from_preview = decode(path, full=full)
    rec["from_preview"] = from_preview
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    rec["decoded"] = f"{W}x{H}"

    sc = DET_W / max(H, W)
    small = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else gray
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR))
    faces = [] if faces is None else faces
    rec["faces"] = len(faces)

    inv = 1.0 / sc
    subject_label = "center"
    if len(faces):
        # YuNet 已经返回了所有人脸, 以前这里一行 max(面积) 就把其他的全丢了 ——
        # 而合影里主角的脸往往不是最大的 (大的是前排路人), 于是系统性选错主体。
        # 现在逐张脸算眼部锐度, 取最清楚的那张当主体;
        # 用户也可以在界面上手动换 (见 Api.set_subject)。
        best, all_face_boxes, face_metrics = pick_subject(gray, faces, inv, W, H,
                                                          bgr=bgr, lmk=_W.get("lmk"))
        eye_sharp, fx, fy, fw, fh = best["eye"], best["fx"], best["fy"], best["fw"], best["fh"]
        lms, conf, er, eboxes = (best["lms"], best["conf"], best["er"],
                                 best["eye_boxes"])
        face_ten, face_std = patch_sharpness(gray, fx + fw / 2, fy + fh / 2, max(fw, fh) * 0.55)

        rec.update({
            "eye_sharp": round(eye_sharp, 1),
            "face_sharp": round(face_ten, 1),
            "face_conf": round(conf, 2),
            "face_w": int(fw),
            "face_cx": round((fx + fw / 2) / W, 4),
            "face_cy": round((fy + fh / 2) / H, 4),
            "face_area_pct": round(fw * fh / (W * H) * 100, 2),
            # sig 必须是**当前主体那张脸**的指纹, 否则分组会拿 A 图的第 3 张脸
            # 去和 B 图的第 1 张脸比, 主体就漂了
            "sig": face_signature(gray, (fx, fy, fw, fh)),
            "_face_box": [int(fx), int(fy), int(fw), int(fh)],
            "_eye_boxes": eboxes,
            "_face_boxes": all_face_boxes,
            "_face_metrics": face_metrics,
            "subject_idx": best["idx"],
            "yaw": round(best["yaw"], 1) if best.get("yaw") is not None else None,
            "pitch": round(best["pitch"], 1) if best.get("pitch") is not None else None,
            "roll": round(best["roll"], 1) if best.get("roll") is not None else None,
            "ear": round(best["ear"], 3) if best.get("ear") is not None else None,
        })
        subject_label = "face"
        rec["compare_value"] = rec["eye_sharp"]
    else:
        t, c = patch_sharpness(gray, W * 0.5, H * 0.45, min(W, H) * 0.22)
        rec.update({"eye_sharp": 0.0, "face_sharp": round(t, 1), "face_conf": 0.0,
                    "face_w": 0, "face_cx": 0.5, "face_cy": 0.45, "face_area_pct": 0,
                    "sig": None, "_face_box": None, "_eye_boxes": [],
                    "_face_boxes": [], "_face_metrics": [], "subject_idx": -1,
                    "yaw": None, "pitch": None, "roll": None, "ear": None})
        # 和眼部一样除以对比度, 否则无脸照片的 compare_value 还是原始梯度能量,
        # 量级比归一化后的眼部锐度大一个数量级, 排序时会全被顶到最前面
        rec["compare_value"] = round(t / (c + 1.0), 1)

    rec["subject"] = subject_label

    # 技术画质 (MUSIQ 0~100) / 美感 (NIMA 1~10) —— 整图, 每张各跑一次。
    # 模型缺失就留 None, 后面的子分数会按"该维度不参与"处理。
    rec["mos_tech"] = None
    rec["aes"] = None
    if _W.get("musiq") is not None:
        try:
            rec["mos_tech"] = round(tech_quality(bgr, _W["musiq"]), 2)
        except Exception:                                   # noqa: BLE001
            pass
    if _W.get("nima") is not None:
        try:
            rec["aes"] = round(aesthetic_score(bgr, _W["nima"]), 2)
        except Exception:                                   # noqa: BLE001
            pass

    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    rec["sharp_global"] = round(float(tenengrad_map(g).mean()), 1)
    rec["brightness"] = round(float(gray.mean()), 1)
    rec["clipped"] = round(float(((gray > 250).sum() + (gray < 5).sum()) / gray.size * 100), 2)
    rec["_thumb"] = make_thumb(bgr, rec["_face_box"], rec["_eye_boxes"], af_point,
                               all_face_boxes=rec.get("_face_boxes"))
    rec["preview_rel"] = ""
    if preview_path:
        try:
            pv = bgr if max(bgr.shape[:2]) <= preview_side else cv2.resize(
                bgr, None, fx=preview_side / max(bgr.shape[:2]), fy=preview_side / max(bgr.shape[:2]),
                interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", pv, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with open(preview_path, "wb") as fh:
                    fh.write(buf.tobytes())
                rec["preview_rel"] = "preview/" + os.path.basename(preview_path)
        except Exception:                                   # noqa: BLE001
            pass
    rec["_ms"] = int((time.time() - t0) * 1000)
    return rec


def make_thumb(bgr, face_box, eye_boxes, af_point=None, max_w=360, quality=78,
               all_face_boxes=None):
    """用 cv2 缩略图 + 画框 (比 PIL LANCZOS 快数倍, 实测 84ms -> ~15ms)。

    all_face_boxes 给了就把检出的**所有**脸都画出来 (暗红细框), 选中的那张
    用亮红粗框 —— 界面上就能一眼看出"这张检出 8 张脸, 用的是哪一张"。
    """
    h, w = bgr.shape[:2]
    k = max_w / w
    im = cv2.resize(bgr, (max_w, max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    sel = [int(round(v)) for v in face_box] if face_box else None
    for fb in (all_face_boxes or []):
        x, y, bw, bh = [int(round(v * k)) for v in fb]
        if sel and [int(round(v)) for v in fb] == sel:
            continue                      # 选中那张最后画粗框
        cv2.rectangle(im, (x, y), (x + bw, y + bh), (60, 40, 150), 1)   # 暗红 = 其他脸
    if sel:
        x, y, bw, bh = [int(round(v * k)) for v in sel]
        cv2.rectangle(im, (x, y), (x + bw, y + bh), (64, 96, 255), 2)   # 亮红 = 主体
    for x, y, bw, bh in (eye_boxes or []):
        x, y, bw, bh = [int(round(v * k)) for v in (x, y, bw, bh)]
        cv2.rectangle(im, (x, y), (x + bw, y + bh), (255, 210, 0), 1)   # 青 = 眼睛区域
    if af_point:                                                            # 绿 = 相机设定的对焦点
        ax, ay = int(round(af_point[0] * im.shape[1])), int(round(af_point[1] * im.shape[0]))
        cv2.circle(im, (ax, ay), 13, (0, 220, 0), 2)
        cv2.drawMarker(im, (ax, ay), (0, 220, 0), cv2.MARKER_CROSS, 17, 2)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ==========================================================================
# 分组: 同一时刻的连拍才算可比 (时间窗 + 姿态)
# ==========================================================================
def ncc(a, b):
    return float(np.dot(a, b) / a.size)


# 连拍窗口: 拍摄时间相隔不超过这么多秒的帧才可能归为一组。
# 以前是**全局**两两比对 + 并查集传递闭包, "甲像乙、乙像丙"的链会一路滚雪球,
# 实测把 2 小时里 254 张互不相关的照片并成一组 (主体脸中心横跨 0.34~0.68)。
# 连拍本就是一串相隔几秒的照片, 把候选配对限制在时间邻域内, 既贴合作品语义,
# 也天然掐断了跨时段的滚雪球。
BURST_SECS = 30.0
# 窗口内、位置和脸宽都吻合时允许的指纹下限。人走动/转头会让整脸指纹掉得很低
# (实测同一段走位里相邻帧 NCC 只有 0.05), 这时主要靠"时间近 + 位置/脸宽吻合"
# 判定同组, 指纹只用来挡住完全不相关的帧。
BURST_NCC = 0.20
# 连拍判据用的位置/脸宽容差: 比常规判据更严, 因为这里放宽了指纹门槛。
BURST_POS_TOL = 0.06
BURST_SIZE_TOL = 0.20


def _row_time(r):
    """EXIF 拍摄时间 -> datetime; 缺失或非法返回 None (退回按文件顺序近似)。"""
    s = r.get("datetime_original")
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _same_pose(a, b, pos_tol, size_tol, ncc_min, focal_tol, iou_min=0.0):
    """两张照片的主体是不是"同一个人的相近姿态"。几道门槛全过才算。

    iou_min 是"主体脸框"的重叠下限 —— 位置差只管脸中心, 但两张照片里
    主体压根不是同一个人时, 脸中心照样可能靠得很近 (同一排座位上的邻座)。
    IoU 补的就是这个洞。
    """
    if abs(math.log2(max(a["face_w"], 1) / max(b["face_w"], 1))) > size_tol:
        return False
    if math.hypot(a["face_cx"] - b["face_cx"], a["face_cy"] - b["face_cy"]) > pos_tol:
        return False
    fr, fp = a.get("focal_length"), b.get("focal_length")
    if fr and fp and abs(math.log2((fr[0] / fr[1]) / (fp[0] / fp[1]))) > focal_tol:
        return False
    if iou_min > 0 and _face_iou_norm(a, b) < iou_min:
        return False
    return ncc(a["sig"], b["sig"]) >= ncc_min


def _burst_same(a, b, pos_tol, size_tol, ncc_min):
    """连拍窗口内的宽松判据: 位置/脸宽严格吻合 + 指纹不低于下限。

    整脸指纹对走动/转头很敏感 (实测同一段走位里相邻帧能掉到 0.05), 只在
    "时间很近 + 位置几乎没动 + 脸一样大"时才放宽指纹门槛, 把同一串连拍收进
    一组, 同时仍用指纹下限挡住画面里换人/换背景的情况。
    """
    if abs(math.log2(max(a["face_w"], 1) / max(b["face_w"], 1))) > size_tol:
        return False
    if math.hypot(a["face_cx"] - b["face_cx"], a["face_cy"] - b["face_cy"]) > pos_tol:
        return False
    return ncc(a["sig"], b["sig"]) >= ncc_min


def group_frames(rows, pos_tol=0.12, size_tol=0.30, ncc_min=0.85, focal_tol=0.15,
                 iou_min=0.0, burst_secs=BURST_SECS, burst_ncc=BURST_NCC):
    """把「同一时刻、同一个人的相近姿态」的帧归为一组, 只在组内比锐度。

    并查集 (传递闭包), 但**候选配对被时间窗限制**: 只有拍摄时间相隔 <=
    burst_secs 的帧才两两比对。这道限制是必需的 —— 纯全局闭包会滚雪球,
    实测把 2 小时里 254 张不相关的照片并成一组。时间窗把分组锁在"连拍"邻域内。

    每对被时间窗允许的帧, 判定分两级:
      1. `_same_pose` 常规判据: 位置/脸宽/焦距/指纹 (ncc_min) 全过
      2. `_burst_same` 连拍判据: 时间很近时位置/脸宽几乎没变就放宽指纹门槛,
         用来把"人走动/转头导致整脸指纹掉下来"的同一串连拍收在一起

    没有 EXIF 拍摄时间的帧退回按文件顺序的相邻 (|i-j|<=3) 兜底。

    ncc_min 仍是实测分界 (一批 25 张会议/合影照逐对量出来 0.85: 正确的对
    NCC=0.905, 错误的对是 0.847/0.633, 位置差和 IoU 全都分不开)。但那批数据
    里每对都是"站着几乎不动", 一旦有人走动/转头整脸指纹就会掉到 0.2 以下,
    这时要靠第 2 级判据兜底 —— ncc_min 是主判据, 不再是唯一判据。
    pos_tol 保留 0.12: 它挡的是"同一画面里不同的人", 仍是最有效的一道闸。
    """
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]        # 路径压缩
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    times = [_row_time(r) for r in rows]
    idx = [i for i, r in enumerate(rows)
           if r.get("subject") == "face" and r.get("sig") is not None]
    for i, j in itertools.combinations(idx, 2):
        a, b = rows[i], rows[j]
        ta, tb = times[i], times[j]
        if ta is not None and tb is not None:
            if abs((ta - tb).total_seconds()) > burst_secs:
                continue                          # 相隔太久, 不可能是同一串连拍
            if _same_pose(a, b, pos_tol, size_tol, ncc_min, focal_tol, iou_min):
                union(i, j)
            elif _burst_same(a, b, BURST_POS_TOL, BURST_SIZE_TOL, burst_ncc):
                union(i, j)
        elif abs(i - j) <= 3:                     # 没时间戳, 退回按文件顺序相邻
            if _same_pose(a, b, pos_tol, size_tol, ncc_min, focal_tol, iou_min):
                union(i, j)

    buckets = {}
    for i, r in enumerate(rows):
        buckets.setdefault(find(i), []).append(r)
    return list(buckets.values())


def _face_iou_norm(a, b):
    """两张照片里"主体脸框"的交并比。坐标要归一化 ——
    _face_box 是像素, 而不同照片分辨率不一样, 直接比框面积毫无意义。"""
    try:
        aw, ah = [float(x) for x in a["decoded"].split("x")]     # 注意是 WxH
        bw, bh = [float(x) for x in b["decoded"].split("x")]
        ax, ay, aw_, ah_ = [float(v) for v in a["_face_box"]]
        bx, by, bw_, bh_ = [float(v) for v in b["_face_box"]]
    except Exception:                                            # noqa: BLE001
        return 1.0
    if not (aw and ah and bw and bh):
        return 1.0
    ax, ay, aw_, ah_ = ax / aw, ay / ah, aw_ / aw, ah_ / ah
    bx, by, bw_, bh_ = bx / bw, by / bh, bw_ / bw, bh_ / bh
    iw = min(ax + aw_, bx + bw_) - max(ax, bx)
    ih = min(ay + ah_, by + bh_) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    smaller = min(aw_ * ah_, bw_ * bh_)
    return (inter / smaller) if smaller else 0.0


def _subject_may_differ(group):
    """这个组里"选中的那张脸"可能不是同一个人。

    背景: 多脸图里我们取**最清楚**的那张脸当主体 (见 analyze())。但实测一批
    合影, 头部锐度和次头部常常只差 1.0~1.3 倍, 也就是说主体很容易在人之间
    跳 —— 一张选中前排路人, 下一张选中台上的人。这种组里的"眼部锐度"
    不是同一个人的, 组内最佳/疑似模糊就不可信。

    判据: 组内选中的脸框两两几乎不重叠 -> 多半换人了。

    只用 IoU 这一条, 不加"多脸图过半就报"那种粗判: 实测一批 25 张合影,
    IoU=0.79 (人工核对是同一个人) 的组也会被"多脸过半"误报, 而多脸本身
    根本不说明主体会跳 —— 主体跳不跳取决于组内**各自选中了谁**, IoU 才是
    直接证据。
    """
    if len(group) < 2:
        return False
    for a, b in itertools.combinations(group, 2):
        if _face_iou_norm(a, b) < 0.30:
            return True
    return False


# ==========================================================================
# 挑片与移动
# ==========================================================================
SIDECAR_EXTS = (".xmp", ".acr")        # 跟照片一起走的编辑记录 (Lightroom/ACR 的修改就存在这里)


def select_rows(rows, best_per_group=False, min_ratio=0.0, min_sharp=0.0):
    """按条件挑出要保留的照片。

    这里刻意保持"保守"的语义, 免得误删:
    - best_per_group: 每组只留最锐的那张; **单张照片照原样保留**
      (它本来就是自己那组唯一的一张, 没有"输给同伴"的问题)
    - min_ratio: 只对多帧相似组有意义, 单张视为通过, 否则一次拍摄里绝大多数单张都会被丢掉
    """
    out = []
    for r in rows:
        if r.get("_error"):
            continue
        if best_per_group and r.get("group_size", 1) > 1 and not r.get("best_in_group"):
            continue
        if min_ratio > 0 and r.get("group_size", 1) > 1 and r.get("ratio_group", 0) < min_ratio:
            continue
        if min_sharp > 0 and r.get("compare_value", 0) < min_sharp:
            continue
        out.append(r)
    return out


def parse_name_list(path, valid_exts):
    """从清单文件里提取文件名。

    容忍两种格式: HTML 导出的纯文件名一行一个,
    或者 各组最佳.txt 那种 "文件名<tab>锐度<tab>..." 的行。
    """
    names = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            for tok in line.strip().split():
                tok = tok.strip().strip('"')
                if os.path.splitext(tok)[1].lower() in valid_exts:
                    names.append(os.path.basename(tok))
                    break
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def plan_move(folder, names, dest, with_jpg=False):
    """算出要移动哪些文件。返回 (可执行动作, 跳过说明)"""
    todo, skipped = [], []
    for n in names:
        src = os.path.join(folder, n)
        if not os.path.isfile(src):
            skipped.append((n, "源文件不存在"))
            continue
        stem = os.path.splitext(n)[0]
        targets = [n] + [stem + e for e in SIDECAR_EXTS]
        if with_jpg:
            targets += [stem + ".jpg", stem + ".jpeg"]
        for t in targets:
            s = os.path.join(folder, t)
            if not os.path.isfile(s):
                continue
            d = os.path.join(dest, t)
            if os.path.exists(d):
                skipped.append((t, "目标已存在, 为防覆盖已跳过"))
                continue
            todo.append((s, d))
    return todo, skipped


def do_move(folder, names, dest, copy=False, with_jpg=False, dry_run=False, out_dir=None):
    folder = os.path.abspath(folder)
    dest = os.path.abspath(dest)
    if dest == folder:
        print("  [!] 目标文件夹和照片文件夹相同, 已取消。")
        return 0, 0
    # commonpath 在两个盘符之间会抛 ValueError (比如 F:\照片 -> C:\精选),
    # 所以先按盘符分一下, 跨盘就直接跳过"目标在源文件夹内部"这个提示
    if os.path.splitdrive(folder)[0].lower() == os.path.splitdrive(dest)[0].lower():
        try:
            if os.path.commonpath([folder, dest]) == folder and dest != folder:
                print(f"  [!] 提示: 目标在照片文件夹内部 ({dest})")
        except ValueError:
            pass

    todo, skipped = plan_move(folder, names, dest, with_jpg)
    print(f"\n{'【试运行】' if dry_run else ''}共 {len(names)} 张, "
          f"涉及 {len(todo)} 个文件{' (含 .xmp/.acr 编辑记录)' if any(t[0].lower().endswith(SIDECAR_EXTS) for t in todo) else ''}")

    if not dry_run and todo:
        os.makedirs(dest, exist_ok=True)

    moved = failed = 0
    log_lines = [f"{'复制' if copy else '移动'}到: {dest}", ""]
    for s, d in todo:
        if dry_run:
            log_lines.append(f"[试运行] {os.path.basename(s)}")
            moved += 1
            continue
        try:
            if copy:
                shutil.copy2(s, d)
            else:
                shutil.move(s, d)
            log_lines.append(os.path.basename(s))
            moved += 1
        except Exception as exc:                            # noqa: BLE001
            log_lines.append(f"!! {os.path.basename(s)}  失败: {exc}")
            failed += 1

    for name, why in skipped:
        log_lines.append(f"跳过 {name} — {why}")

    print(f"  {'将' if dry_run else '已'}{'复制' if copy else '移动'} {moved} 个文件"
          f"{f', 失败 {failed} 个' if failed else ''}"
          f"{f', 跳过 {len(skipped)} 个' if skipped else ''}")
    for name, why in skipped[:8]:
        print(f"    跳过 {name} — {why}")
    if len(skipped) > 8:
        print(f"    ... 另有 {len(skipped) - 8} 个, 见日志")

    if out_dir and not dry_run:
        try:
            os.makedirs(out_dir, exist_ok=True)
            lp = os.path.join(out_dir, "移动日志.txt")
            with open(lp, "a", encoding="utf-8") as fh:
                fh.write("\n".join(log_lines) + "\n" + "-" * 50 + "\n")
            print(f"  日志: {lp}")
        except OSError as exc:
            print(f"  [!] 日志写不进去 ({exc}), 但文件已经处理完了")
    return moved, failed


# ==========================================================================
# 报告
# ==========================================================================
HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>选图报告</title>
<style>
 body{background:#18181b;color:#e9e9ec;font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;margin:0;padding:22px}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:#8f8f99;margin-bottom:14px;font-size:12.5px}
 .stat{display:inline-block;background:#242429;border-radius:8px;padding:7px 13px;margin:0 7px 7px 0;font-size:12.5px}
 .stat b{font-size:17px}
 .bar{background:#202024;border:1px solid #303038;border-radius:10px;padding:12px 14px;margin:0 0 14px;position:sticky;top:0;z-index:20}
 .bar .row{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
 .bar label{font-size:12.5px;color:#c9c9d2;display:flex;align-items:center;gap:5px}
 input[type=number],input[type=text],select{background:#2a2a31;color:#eee;border:1px solid #3d3d46;border-radius:6px;padding:4px 7px;font:inherit;font-size:12.5px}
 input[type=number]{width:70px}
 input[type=text]{width:280px}
 button{background:#33333c;color:#e9e9ec;border:1px solid #46464f;border-radius:6px;padding:5px 11px;font:inherit;font-size:12.5px;cursor:pointer}
 button:hover{background:#3d3d47}
 button.pri{background:#27ae60;border-color:#27ae60;color:#04220f;font-weight:600}
 button.pri:hover{background:#2ec26c}
 textarea{width:100%;height:52px;background:#17171a;color:#b9e6c9;border:1px solid #3d3d46;border-radius:6px;padding:7px;font:12px/1.5 Consolas,monospace;margin-top:8px;resize:vertical}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;align-items:start}
 .card{background:#242429;border-radius:10px;overflow:hidden;border:2px solid transparent;position:relative}
 .card img{width:100%;display:block}
 .card.soft{border-color:#c0392b}
 .card.best{border-color:#27ae60}
 .card.sel{border-color:#2f7fd8}
 .pick{position:absolute;top:7px;left:7px;z-index:3;background:rgba(0,0,0,.55);border-radius:5px;padding:2px 5px}
 .pick input{width:16px;height:16px;vertical-align:middle;cursor:pointer}
 .info{padding:8px 10px;font-size:12.5px}
 .name{font-weight:600;font-size:13px}
 .sc{color:#7fd1a8}
 .card.soft .sc{color:#ff8a80}
 .meta{color:#8f8f99;font-size:11.5px;margin-top:3px}
 .badge{float:right;font-size:11px;padding:1px 7px;border-radius:99px;background:#3a3a42;color:#bbb}
 .card.soft .badge{background:#c0392b;color:#fff}
 .card.best .badge{background:#27ae60;color:#062}
 .legend,.note{color:#8f8f99;font-size:12px;line-height:1.9}
 .legend{margin:10px 0 16px}
 .note{margin-top:10px}
 .sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:4px}
 a.lb{display:block;cursor:zoom-in}
 #lb{display:none;position:fixed;inset:0;background:rgba(8,8,10,.94);z-index:100;flex-direction:column;align-items:center;justify-content:center;padding:18px}
 #lb img{max-width:96vw;max-height:82vh;object-fit:contain;background:#000}
 #lb .lt{margin-top:10px;font-size:13px;color:#ddd}
 #lb .lt a{color:#7fd1a8;margin-left:12px}
 #lb .hint{color:#77777f;font-size:12px;margin-top:4px}
 .nav{position:absolute;top:50%;transform:translateY(-50%);font-size:34px;color:#9a9aa2;cursor:pointer;user-select:none;padding:10px 18px}
 .nav:hover{color:#fff}
 #navL{left:6px} #navR{right:6px}
 code{background:#242429;padding:1px 5px;border-radius:4px;color:#b9e6c9}
</style></head><body>
<h1>选图报告</h1>
<div class="sub">@@SUB@@</div>
<div>@@STATS@@</div>
<div class="legend">@@LEGEND@@</div>

<div class="bar">
  <div class="row">
    <label>眼部锐度 &ge; <input type="number" id="fSharp" step="50" placeholder="不限"></label>
    <label>组内占比 &ge; <input type="number" id="fRatio" min="0" max="100" step="5" placeholder="不限"> %</label>
    <label><input type="checkbox" id="fBest"> 只看组内胜出</label>
    <label><input type="checkbox" id="fMulti"> 只看有可比对对象的组</label>
    <label>对焦依据
      <select id="fNface"><option value="">全部</option><option value="1">有人脸</option><option value="0">无脸</option></select>
    </label>
    <label>AF区域
      <select id="fArea"><option value="">全部</option>@@AREAS@@</select>
    </label>
    <button id="bReset">重置</button>
  </div>
  <div class="row" style="margin-top:10px">
    <span class="stat" style="margin:0">显示 <b id="vcount">0</b> / @@TOTAL@@ 张</span>
    <span class="stat" style="margin:0">已勾选 <b id="scount">0</b> 张</span>
    <button id="bAll">全选可见</button>
    <button id="bNone">清空</button>
    <button id="bInv">反选可见</button>
    <button class="pri" id="bExport">导出勾选清单</button>
  </div>
  <div class="note">
    浏览器不能直接移动硬盘上的文件(沙箱限制)。所以流程是: <b>勾选</b> → <b>导出勾选清单</b> →
    把下面命令里的目标文件夹填好, 复制到 cmd 窗口执行即可移动。<br>
    目标文件夹: <input type="text" id="dest" placeholder="D:\照片\精选">
    <br><textarea id="cmd" readonly spellcheck="false"></textarea>
    <button id="bCopy" style="margin-top:6px">复制命令</button>
    <span id="copyMsg" class="note"></span>
  </div>
</div>

<div class="grid">
@@CARDS@@
</div>

<div id="lb">
  <span class="nav" id="navL">&#10094;</span>
  <img alt="">
  <div class="lt"><span id="lbName"></span><a id="lbOrig" href="#" target="_blank">在原文件夹中打开原文件</a></div>
  <div class="hint">&#8592;/&#8594; 切换 &nbsp;&middot;&nbsp; Esc 关闭</div>
  <span class="nav" id="navR">&#10095;</span>
</div>

<script>
var cards = Array.prototype.slice.call(document.querySelectorAll('.card'));
var lb = document.getElementById('lb');
var lbImg = lb.querySelector('img');
var curLink = null;

function visibleCards(){ return cards.filter(function(c){ return c.style.display !== 'none'; }); }

function applyFilter(){
  var sharp = document.getElementById('fSharp').value;
  var ratio = document.getElementById('fRatio').value;
  var best  = document.getElementById('fBest').checked;
  var multi = document.getElementById('fMulti').checked;
  var nface = document.getElementById('fNface').value;
  var area  = document.getElementById('fArea').value;
  cards.forEach(function(c){
    var d = c.dataset, ok = true;
    if (sharp !== '' && parseFloat(d.sharp) < parseFloat(sharp)) ok = false;
    if (ok && ratio !== '' && parseFloat(d.ratio) * 100 < parseFloat(ratio)) ok = false;
    if (ok && best && d.best !== '1') ok = false;
    if (ok && multi && d.multi !== '1') ok = false;
    if (ok && nface === '1' && d.face !== '1') ok = false;
    if (ok && nface === '0' && d.face !== '0') ok = false;
    if (ok && area !== '' && d.area !== area) ok = false;
    c.style.display = ok ? '' : 'none';
  });
  document.getElementById('vcount').textContent = visibleCards().length;
  updateCmd();
}

function checkedBoxes(){
  return Array.prototype.slice.call(document.querySelectorAll('.cb')).filter(function(b){ return b.checked; });
}

function updateCount(){
  var n = checkedBoxes().length;
  document.getElementById('scount').textContent = n;
  document.querySelectorAll('.card').forEach(function(c){
    var cb = c.querySelector('.cb');
    if (cb) c.classList.toggle('sel', cb.checked);
  });
  updateCmd();
}

function updateCmd(){
  var names = checkedBoxes().map(function(b){ return b.value; });
  var dest = document.getElementById('dest').value || '<目标文件夹>';
  var txt = '';
  if (names.length) {
    txt = '@@BAT@@ @@FOLDERQ@@' +
          ' --keep-from @@OUTDIRQ@@' +
          ' --keep-to "' + dest + '"';
  }
  var ta = document.getElementById('cmd');
  ta.value = txt;
}

document.querySelectorAll('.bar input, .bar select').forEach(function(el){
  el.addEventListener('input', applyFilter);
  el.addEventListener('change', applyFilter);
});
document.getElementById('dest').addEventListener('input', updateCmd);

document.getElementById('bReset').onclick = function(){
  document.getElementById('fSharp').value = '';
  document.getElementById('fRatio').value = '';
  document.getElementById('fBest').checked = false;
  document.getElementById('fMulti').checked = false;
  document.getElementById('fNface').value = '';
  document.getElementById('fArea').value = '';
  applyFilter();
};
document.getElementById('bAll').onclick = function(){
  visibleCards().forEach(function(c){ var b = c.querySelector('.cb'); if (b) b.checked = true; });
  updateCount();
};
document.getElementById('bNone').onclick = function(){
  document.querySelectorAll('.cb').forEach(function(b){ b.checked = false; });
  updateCount();
};
document.getElementById('bInv').onclick = function(){
  visibleCards().forEach(function(c){ var b = c.querySelector('.cb'); if (b) b.checked = !b.checked; });
  updateCount();
};

document.querySelectorAll('.cb').forEach(function(b){ b.addEventListener('change', updateCount); });

document.getElementById('bExport').onclick = function(){
  var names = checkedBoxes().map(function(b){ return b.value; });
  var msg = document.getElementById('copyMsg');
  if (!names.length) { msg.textContent = '还没有勾选任何照片。'; return; }
  var blob = new Blob([names.join('\r\n') + '\r\n'], {type:'text/plain;charset=utf-8'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '\u9009\u56fe\u6e05\u5355.txt';
  document.body.appendChild(a); a.click(); a.remove();
  msg.textContent = '已导出 ' + names.length + ' 个文件名 (在浏览器的下载目录里)。';
};

document.getElementById('bCopy').onclick = function(){
  var ta = document.getElementById('cmd'), msg = document.getElementById('copyMsg');
  if (!ta.value) { msg.textContent = '先勾选照片。'; return; }
  ta.select();
  try {
    navigator.clipboard.writeText(ta.value).then(function(){ msg.textContent = '已复制。'; },
      function(){ document.execCommand('copy'); msg.textContent = '已复制(兼容模式)。'; });
  } catch (e) { document.execCommand('copy'); msg.textContent = '已复制(兼容模式)。'; }
};

/* ---- 点缩略图看大图 ---- */
document.querySelectorAll('a.lb').forEach(function(a){
  a.addEventListener('click', function(e){ e.preventDefault(); show(a); });
});
function show(a){
  curLink = a;
  lbImg.src = a.getAttribute('href');
  document.getElementById('lbName').textContent = a.dataset.name;
  document.getElementById('lbOrig').href = a.dataset.orig;
  lb.style.display = 'flex';
}
function nav(d){
  if (!curLink) return;
  var list = Array.prototype.slice.call(document.querySelectorAll('a.lb'))
    .filter(function(x){ return x.closest('.card').style.display !== 'none'; });
  var i = list.indexOf(curLink);
  if (i < 0) i = 0;
  show(list[(i + d + list.length) % list.length]);
}
document.getElementById('navL').onclick = function(){ nav(-1); };
document.getElementById('navR').onclick = function(){ nav(1); };
lb.addEventListener('click', function(e){ if (e.target === lb) lb.style.display = 'none'; });
document.addEventListener('keydown', function(e){
  if (lb.style.display !== 'flex') return;
  if (e.key === 'Escape') lb.style.display = 'none';
  else if (e.key === 'ArrowRight') nav(1);
  else if (e.key === 'ArrowLeft') nav(-1);
});

applyFilter();
updateCount();
</script>
</body></html>
"""


BANNER = """============================================================
  选图工具 - 粗筛合焦照片
============================================================

  把【照片文件夹】直接拖到这个窗口里, 然后按回车。
  (也可以在里面粘贴路径; 一次一个文件夹; 输入 q 退出)

"""


def ask_folder():
    """交互式要一个目录: 支持把文件夹拖进 cmd 窗口。"""
    for _ in range(5):
        try:
            raw = input("  文件夹: ")
        except (EOFError, KeyboardInterrupt):
            return None
        p = raw.strip().strip('"').strip("'").strip()
        if p.lower() in ("q", "quit", "exit"):
            return None
        if not p:
            continue
        # 拖拽过来的路径可能带引号、结尾反斜杠、或前后有空格
        p = os.path.expandvars(os.path.expanduser(p))
        p = os.path.abspath(p)
        if os.path.isdir(p):
            return p
        print(f"  [!] 找不到这个文件夹: {p}\n      再试一次, 或输入 q 退出。\n")
    return None


def js_str_body(s):
    """把字符串转成可安全塞进 JS 单引号字面量里的内容。

    用 json.dumps 做转义, 但要剥掉它自带的外层引号, 否则引号会翻倍。
    """
    return json.dumps(s, ensure_ascii=False)[1:-1]


# ==========================================================================
# 分组 + 生成全部报告文件 (命令行 main() 和 GUI app.py 共用)
# ==========================================================================
def _score_parts(r):
    """把一张照片的原始测量折算成 0~1 的子分数 (缺的维度不放进来)。"""
    p = {}
    gs = r.get("group_size", 1)
    cv = float(r.get("compare_value") or 0.0)
    if gs > 1 and r.get("ratio_group"):
        p["sharp"] = max(0.0, min(1.0, float(r["ratio_group"])))
    else:
        p["sharp"] = max(0.0, min(1.0, cv / 25.0))       # 单张: 绝对锐度的饱和映射
    mt = r.get("mos_tech")
    if mt is not None:
        p["tech"] = max(0.0, min(1.0, float(mt) / 100.0))
    yaw = r.get("yaw")
    if yaw is not None:
        ypen = max(0.0, (abs(float(yaw)) - 15.0) / 45.0)
        ppen = max(0.0, (abs(float(r.get("pitch") or 0.0)) - 15.0) / 35.0)
        p["pose"] = max(0.0, 1.0 - max(ypen, ppen))
    ear = r.get("ear")
    if ear is not None:
        p["expr"] = max(0.0, min(1.0, (float(ear) - 0.08) / 0.17))   # <0.08 视为闭眼
    aes = r.get("aes")
    if aes is not None:
        p["aes"] = max(0.0, min(1.0, (float(aes) - 1.0) / 9.0))
    return p


def edit_value_of(parts):
    """按 _CFG 权重把子分数加权合成 0~100; 权重只在"有的维度"上归一化。"""
    w = _CFG.get("weights", {})
    num = den = 0.0
    for k, s in parts.items():
        wk = float(w.get(k, 0.0))
        if wk > 0:
            num += wk * s
            den += wk
    return round(100.0 * num / den, 1) if den > 0 else None


def compute_flags(rows, groups_ncc=0.85, soft_ratio=0.55, group_window=BURST_SECS):
    """给 rows 打上分组/判定标记。纯计算, 不碰磁盘。

    GUI 直接调这个函数出结果, 不经过 write_reports —— 所以 exe 里不用写任何文件。
    """
    groups = group_frames([r for r in rows if not r.get("_error")],
                          ncc_min=groups_ncc, burst_secs=group_window)
    for gi, g in enumerate(groups, 1):
        g.sort(key=lambda r: -r["compare_value"])
        top = g[0]["compare_value"]
        for r in g:
            r["group"] = gi
            r["group_size"] = len(g)
            r["ratio_group"] = round(r["compare_value"] / top, 3) if top else 0
            r["best_in_group"] = False
            r["soft"] = bool(top and len(g) > 1 and r["compare_value"] < top * soft_ratio)
        if len(g) > 1:
            g[0]["best_in_group"] = True
        # 主体可能不是同一个人 —— 见 _subject_may_differ 的说明。
        # 命中就在界面上标出来提醒, 别让用户误以为组内锐度可以直接比。
        risky = len(g) > 1 and _subject_may_differ(g)
        for r in g:
            r["subject_uncertain"] = risky
    for r in rows:
        r.setdefault("group", 0)
        r.setdefault("group_size", 1)
        r.setdefault("ratio_group", 0)
        r.setdefault("best_in_group", False)
        r.setdefault("soft", bool(r.get("_error")))
        r.setdefault("subject_uncertain", False)

    # 修图价值: 各维度子分数 (0~1) + 加权合成 (0~100)
    for r in rows:
        parts = _score_parts(r)
        r["score_parts"] = {k: round(v, 3) for k, v in parts.items()}
        r["edit_value"] = None if r.get("_error") else edit_value_of(parts)

    return {
        "groups": groups,
        "n_soft": sum(1 for r in rows if r.get("soft")),
        "n_face": sum(1 for r in rows if r.get("faces")),
        "n_dup": sum(1 for g in groups if len(g) > 1),
        "n_uncertain": sum(1 for r in rows if r.get("subject_uncertain")),
    }


CSV_COLS = [("group", "组"), ("group_size", "组内张数"), ("best_in_group", "组内最佳"),
            ("soft", "疑似模糊"), ("edit_value", "修图价值"), ("file", "文件名"),
            ("eye_sharp", "眼部锐度"),
            ("ratio_group", "占组内最佳"), ("yaw", "偏航°"), ("pitch", "俯仰°"),
            ("ear", "睁眼"), ("mos_tech", "技术画质"), ("aes", "美感"),
            ("face_sharp", "脸部锐度"),
            ("sharp_global", "全画面锐度"), ("subject", "对焦依据"), ("face_conf", "人脸置信"),
            ("face_w", "脸宽px"), ("face_area_pct", "脸占画面%"), ("faces", "人脸数"),
            ("focus_mode", "对焦模式"), ("af_area", "AF区域"), ("af_x", "AF点X"), ("af_y", "AF点Y"),
            ("iso", "ISO"), ("exposure_time", "快门"), ("f_number", "光圈"),
            ("focal_length", "焦距"), ("lens_model", "镜头"), ("datetime_original", "拍摄时间"),
            ("brightness", "亮度"), ("clipped", "过曝死黑%"), ("decoded", "解码尺寸"), ("_error", "错误")]


def write_csv(rows, path):
    """只写一个 CSV, 不管其它报告文件。GUI 的「导出 CSV」用这个。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([c[1] for c in CSV_COLS])
        for r in sorted(rows, key=lambda x: (x.get("group") or 999, not x.get("best_in_group"),
                                             -x.get("compare_value", 0))):
            line = []
            for k, _ in CSV_COLS:
                v = r.get(k, "")
                if k == "exposure_time":
                    v = fmt_exposure(v)
                elif k == "f_number":
                    v = fmt_rat(r.get("f_number"), 1)
                elif k == "focal_length":
                    v = f"{fmt_rat(r.get('focal_length'), 0)}mm" if r.get("focal_length") else ""
                elif k == "subject":
                    v = {"face": "眼睛", "center": "中央区(无脸)", "?": "失败"}.get(v, v)
                line.append(v)
            w.writerow(line)


# ==========================================================================
# 分析结果缓存
#
# 目的: 同一批照片第二次点「开始分析」时直接读结果, 不再逐张解码 RAW。
# 为什么不用 CSV 当中转: CSV 写的是**已格式化的字符串** (快门写成 "1/125",
# 对焦依据写成 "眼睛"), 而界面要的是原始类型 (exposure_time 是 (1,125) 元组,
# subject 是 "face"/"center"), 直接读会把界面显示搞坏。
#
# 存的必须是**原始测量值**, 不存 group/soft/best_in_group —— 那几个由
# compute_flags() 纯计算得出, 存了反而会和 --group-ncc/--ratio 参数打架。
# ==========================================================================
CACHE_VERSION = 4          # 改测量逻辑就 +1, 旧缓存自动作废
                            # v2: 选脸规则从"最大面积"改成"最清楚那张", sig 含义跟着变了
                            # v3: 眼部锐度改成对比度归一化 (除以眼部区域 std), 抵消阴影/欠曝
                            # v4: 眼部锐度再按瞳距做尺度归一化 + 按 yaw 做朝向归一化;
                            #     新增 yaw/pitch/roll/ear/mos_tech/aes 与修图价值分
# 缓存就落在**照片文件夹里** (不再写 C 盘), 文件名固定。
# 用固定名而不是"路径的 sha1": 缓存跟照片放一起, 文件夹整体拷贝/挪动后缓存也跟着走 ——
# 但读回时校验 data["folder"] 必须等于当前文件夹, 所以换了路径会作废重扫, 不会串味。
# 后缀是 .json, 不在 RAW_EXTS/IMG_EXTS 白名单里, list_photos() 不会把它当照片。
CACHE_NAME = "_cull_cache.json"

THUMB_PREFIX = "data:image/jpeg;base64,"

# 逐张存的字段。前两个是给过期校验用的, 其余是测量结果。
# _face_box 必须存: _subject_may_differ() 靠它算主体脸框的 IoU 来判断
# "组内选中的脸可能不是同一个人", 不存的话第二次读缓存就再也标不出 ⚠。
_CACHE_FIELDS = ("eye_sharp", "face_sharp", "sharp_global", "compare_value",
                 "subject", "faces", "face_conf", "face_w", "face_cx", "face_cy",
                 "face_area_pct", "brightness", "clipped", "decoded",
                 "from_preview", "_error", "_ms", "_face_boxes", "_face_box",
                 "_eye_boxes", "_face_metrics", "subject_idx",
                 "yaw", "pitch", "roll", "ear", "mos_tech", "aes")
# EXIF: 这些原始值是 (分子, 分母) 元组, JSON 里要转成 list, 读回再转回 tuple
_CACHE_RATIOS = ("exposure_time", "f_number", "focal_length")
_CACHE_PLAIN = ("iso", "focus_mode", "af_area", "af_x", "af_y",
                "lens_model", "datetime_original")


def cache_path(folder):
    """缓存文件 = 照片文件夹里的 _cull_cache.json。"""
    return os.path.join(os.path.abspath(folder), CACHE_NAME)


def _enc_sig(sig):
    """人脸指纹 (64x64 float32) -> base64。分组要靠它做 NCC, 少了就全拆成单张组。"""
    if sig is None:
        return None
    return base64.b64encode(np.asarray(sig, dtype=np.float32).tobytes()).decode("ascii")


def _py(v):
    """numpy 标量/数组 -> 纯 python 类型, 让 json 能序列化。

    不做这一步的话, face_cx/face_cy/face_area_pct 这些由 numpy 算出来的
    np.float32 会让 write_cache 对**每一张**都抛 TypeError 然后静默跳过,
    缓存等于没写。逐个字段转, 顺手也处理 _face_boxes 这种嵌套 list。
    """
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    return v


def _dec_sig(b64):
    if not b64:
        return None
    try:
        # copy() 是必须的: frombuffer 得到的是只读视图, 后面 group_frames 不会改它,
        # 但保持成普通 ndarray 更安全
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()
    except Exception:                                     # noqa: BLE001
        return None


def write_cache(folder, rows, full, groups_ncc=0.85, soft_ratio=0.55):
    """把一轮分析结果落盘。失败不致命 —— 顶多下次重扫, 所以别向上抛。"""
    try:
        items = []
        for r in rows:
            try:
                st = os.stat(os.path.join(folder, r["file"]))
            except OSError:
                continue                    # 分析完又被删掉的, 跳过
            thumb = r.get("_thumb") or ""
            if thumb.startswith(THUMB_PREFIX):
                thumb = thumb[len(THUMB_PREFIX):]
            it = {"f": r["file"], "sz": st.st_size, "mt": int(st.st_mtime),
                  "sig": _enc_sig(r.get("sig")), "thumb": thumb}
            for k in _CACHE_FIELDS:
                it[k] = _py(v) if (v := r.get(k)) is not None else None
            for k in _CACHE_RATIOS:
                v = r.get(k)
                it[k] = list(v) if v else None
            for k in _CACHE_PLAIN:
                it[k] = r.get(k)
            items.append(it)
        data = {"version": CACHE_VERSION, "folder": os.path.abspath(folder),
                "full": bool(full), "groups_ncc": groups_ncc,
                "soft_ratio": soft_ratio, "n": len(items),
                "ts": int(time.time()), "items": items}
        # 先写临时文件再改名, 免得中途被杀写出半个坏文件
        dst = cache_path(folder)
        tmp = dst + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, dst)
        return True
    except Exception as exc:                                # noqa: BLE001
        print(f"  [!] 缓存写不进去 ({exc}), 不影响本次结果")
        return False


def read_cache(folder, full, names=None):
    """读缓存并校验。有效返回 rows (已跑过 compute_flags), 无效返回 None。

    校验项 (任一不满足就当没有缓存):
      - 文件在、能解析、version 对得上
      - full 标志一致 (半分辨率和全分辨率的分数没有可比性)
      - 目录里的照片集合和缓存记录的完全一致 (新增/删了照片)
      - 每张照片现在的大小+修改时间和记录一致 (被改过)
    """
    path = cache_path(folder)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:                                       # noqa: BLE001
        return None
    if data.get("version") != CACHE_VERSION or data.get("folder") != os.path.abspath(folder):
        return None
    if bool(data.get("full")) != bool(full):
        return None

    items = data.get("items") or []
    # 照片集合变了就作废: 否则新拍的那几张根本不在缓存里, 界面上会少照片
    if names is not None:
        if len(names) != len(items) or {n for n in names} != {it.get("f") for it in items}:
            return None

    rows = []
    for it in items:
        fp = os.path.join(folder, it.get("f") or "")
        try:
            st = os.stat(fp)
        except OSError:
            return None                   # 记录里的文件没了
        if st.st_size != it.get("sz") or int(st.st_mtime) != it.get("mt"):
            return None                   # 照片被改过
        r = {"file": it["f"], "orig_uri": pathlib.Path(fp).as_uri(),
             "sig": _dec_sig(it.get("sig")), "preview_rel": ""}
        # 下面 _CACHE_FIELDS 循环会把 _face_box/_eye_boxes/_face_boxes 一起带回来
        r["_face_box"] = it.get("_face_box")
        r["_eye_boxes"] = it.get("_eye_boxes") or []
        thumb = it.get("thumb") or ""
        r["_thumb"] = THUMB_PREFIX + thumb if thumb else ""
        for k in _CACHE_FIELDS:
            r[k] = it.get(k)
        for k in _CACHE_RATIOS:
            v = it.get(k)
            r[k] = tuple(v) if v else None
        for k in _CACHE_PLAIN:
            r[k] = it.get(k)
        rows.append(r)

    compute_flags(rows, groups_ncc=data.get("groups_ncc") or 0.85,
                  soft_ratio=data.get("soft_ratio") or 0.55)
    return rows


def clear_cache(folder):
    """删掉照片文件夹里的缓存文件 (含可能残留的 .tmp)。返回删掉几个。"""
    n = 0
    try:
        for p in (cache_path(folder), cache_path(folder) + ".tmp"):
            if os.path.isfile(p):
                os.remove(p)
                n += 1
    except OSError:
        pass
    return n


def write_reports(rows, folder, out_dir, full=False, groups_ncc=0.85, soft_ratio=0.55,
                  group_window=BURST_SECS):
    """给 rows 打上分组/判定标记, 并写出 CSV / txt / HTML。返回统计数字。"""
    st = compute_flags(rows, groups_ncc=groups_ncc, soft_ratio=soft_ratio,
                       group_window=group_window)
    groups, n_soft, n_face, n_dup = st["groups"], st["n_soft"], st["n_face"], st["n_dup"]
    os.makedirs(out_dir, exist_ok=True)

    # ---- CSV ----
    write_csv(rows, os.path.join(out_dir, "选图报告.csv"))

    # ---- 各组最佳 / 疑似模糊 ----
    multi_g = [g for g in groups if len(g) > 1]
    singles = [r for g in groups if len(g) == 1 for r in g]
    with open(os.path.join(out_dir, "各组最佳.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"共 {len(rows)} 张; {len(multi_g)} 组有可比对的相似帧, {len(singles)} 张是单独的\n\n")
        fh.write("=== 相似帧分组 (组内比较可靠, 已按锐度排序) ===\n\n")
        if not multi_g:
            fh.write("  (这批照片里没有找到几乎同一姿势的帧)\n\n")
        for g in sorted(multi_g, key=lambda g: -len(g)):
            g = sorted(g, key=lambda r: -r["compare_value"])
            fh.write(f'---- 组 {g[0]["group"]}  ({len(g)} 张) ----\n')
            for r in g:
                mark = "   <<< 建议保留(组内最清晰)" if r.get("best_in_group") else (
                    "   偏软" if r.get("soft") else "")
                fh.write(f'  {r["file"]:<16} 眼部锐度 {r["eye_sharp"]:>8}  '
                         f'{r["ratio_group"]:>5.0%}{mark}\n')
            fh.write("\n")
        fh.write("\n=== 单独的照片 (没有可比对对象, 锐度仅供参考) ===\n")
        fh.write("(顺序 = 眼部锐度从高到低; 已按对比度归一化, 但框景大小仍会带偏, 不要跨照片直接比)\n\n")
        for r in sorted(singles, key=lambda r: -r["compare_value"]):
            extra = "  [无脸, 用中央区估算]" if r.get("subject") == "center" else ""
            fh.write(f'  {r["file"]:<16} 眼部锐度 {r["eye_sharp"]:>8}{extra}\n')
    with open(os.path.join(out_dir, "疑似模糊.txt"), "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda x: (x.get("group") or 999, x["compare_value"])):
            if r.get("soft"):
                fh.write(f'{r["file"]}\t组{r.get("group")}({r.get("group_size")}张)\t'
                         f'眼部锐度 {r["eye_sharp"]}\t占组内最佳 {r.get("ratio_group",0):.0%}\n')

    # ---- HTML ----
    areas = sorted({r["af_area"] for r in rows if r.get("af_area")})
    ordered = sorted(rows, key=lambda r: (r.get("subject") != "face", -r.get("compare_value", 0)))
    cards = []
    for r in ordered:
        cls = "card best" if r.get("best_in_group") else ("card soft" if r.get("soft") else "card")
        if r.get("_error"):
            badge = "失败"
        elif r.get("best_in_group"):
            badge = f'组{r["group"]} 最清晰'
        elif r.get("soft"):
            badge = f'组{r["group"]} 疑似模糊'
        else:
            badge = (f'组{r["group"]}' if r.get("group_size", 1) > 1
                     else ("无脸" if r.get("subject") == "center" else ""))
        # 用字面的 · 而不是 &middot;: 下面会 html.escape, 实体里的 & 会被转义成普通文本
        meta = " · ".join(x for x in [
            f'组{r["group"]}({r["group_size"]}张)' if r.get("group_size", 1) > 1 else "",
            "中央区(无脸)" if r.get("subject") == "center" else "眼睛对焦",
            f'脸 {r.get("face_w",0)}px' if r.get("face_w") else "",
            f'AF {r["af_area"]}' if r.get("af_area") else "",
            f'ISO {r["iso"]}' if r.get("iso") else "",
            fmt_exposure(r.get("exposure_time")),
            f'f/{fmt_rat(r.get("f_number"),1)}' if r.get("f_number") else "",
            f'{fmt_rat(r.get("focal_length"),0)}mm' if r.get("focal_length") else "",
        ] if x)
        score = (f'眼部锐度 {r["eye_sharp"]}' if r.get("subject") == "face"
                 else f'中央区锐度 {r.get("face_sharp", 0)}')
        ratio = (f' &nbsp;(占组内最佳 {r["ratio_group"]:.0%})'
                 if r.get("group_size", 1) > 1 and r.get("ratio_group") else "")
        badge_html = f'<span class="badge">{badge}</span>' if badge else ""
        big = html.escape(r.get("preview_rel") or r.get("_thumb") or "", quote=True)
        img = (f'<img loading="lazy" src="{r["_thumb"]}" alt="{html.escape(r["file"])}">'
               if r.get("_thumb") else '<div style="height:200px;background:#333"></div>')
        cards.append(
            f'<div class="{cls}" data-name="{html.escape(r["file"], quote=True)}" '
            f'data-sharp="{r.get("compare_value", 0)}" data-ratio="{r.get("ratio_group", 0)}" '
            f'data-best="{1 if r.get("best_in_group") else 0}" '
            f'data-multi="{1 if r.get("group_size", 1) > 1 else 0}" '
            f'data-face="{1 if r.get("subject") == "face" else 0}" '
            f'data-area="{html.escape(str(r.get("af_area", "")), quote=True)}">'
            f'<div class="pick"><input type="checkbox" class="cb" value="{html.escape(r["file"], quote=True)}"></div>'
            f'<a class="lb" href="{big}" data-name="{html.escape(r["file"], quote=True)}" '
            f'data-orig="{html.escape(r.get("orig_uri", ""), quote=True)}">{img}</a>'
            f'<div class="info">{badge_html}'
            f'<div class="name">{html.escape(r["file"])}</div>'
            f'<div class="sc">{score}{ratio}</div>'
            f'<div class="meta">{html.escape(meta)}</div></div></div>')

    stats = (f'<div class="stat">共 <b>{len(rows)}</b> 张</div>'
             f'<div class="stat">检出人脸 <b>{n_face}</b> 张</div>'
             f'<div class="stat">可对比的相似帧 <b>{n_dup}</b> 组</div>'
             f'<div class="stat">其中疑似模糊 <b style="color:#ff8a80">{n_soft}</b> 张</div>')
    legend = ('<span class="sw" style="background:#27ae60"></span>组内最清晰 &nbsp;&nbsp;'
              '<span class="sw" style="background:#c0392b"></span>组内疑似模糊 &nbsp;&nbsp;'
              '<span class="sw" style="background:#2f7fd8"></span>已勾选 &nbsp;&nbsp;'
              '<span class="sw" style="background:#ff6040"></span>人脸框 &nbsp;&nbsp;'
              '<span class="sw" style="background:#00d2ff"></span>评分用的眼睛区域 &nbsp;&nbsp;'
              '<span class="sw" style="background:#00dc00"></span>相机设定的对焦点(仅 Flexible Spot)<br>'
              '<b>重要</b>: 锐度是眼睛区域的梯度能量÷对比度 (已抵消阴影/欠曝), '
              '但框景大小仍会带偏, <b>只有同一组的帧之间比较才有意义</b>, 不要拿绝对值跨照片比。'
              '列表按锐度从高到低排, 从上往下扫即可。<br>'
              '<b>筛选说明</b>: "组内胜出"指同一组里最锐的那张(单张照片没得比, 不会出现在这个筛选里); '
              '"组内占比"只对多帧相似组有效, 单张视为通过。')
    out_q = out_dir.replace('"', "")
    tool_bat = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.bat")
    out_list = os.path.join(out_q, "选图清单.txt")
    page = (HTML_TEMPLATE
            .replace("@@SUB@@", html.escape(f"{folder} · {len(rows)} 张 · "
                                            f"{'全分辨率' if full else '半分辨率'}解码"))
            .replace("@@STATS@@", stats).replace("@@LEGEND@@", legend)
            .replace("@@AREAS@@", "".join(f'<option value="{html.escape(a, quote=True)}">{a}</option>'
                                          for a in areas))
            .replace("@@TOTAL@@", str(len(rows)))
            .replace("@@CARDS@@", "\n".join(cards))
            # 这几处插进 <script> 当 JS 字符串用, 必须按 JS 规则转义
            .replace("@@FOLDERQ@@", js_str_body('"' + folder + '"'))
            .replace("@@BAT@@", js_str_body('"' + tool_bat + '"'))
            .replace("@@OUTDIRQ@@", js_str_body('"' + out_list + '"')))
    with open(os.path.join(out_dir, "选图报告.html"), "w", encoding="utf-8") as fh:
        fh.write(page)

    return groups, n_soft, n_face, n_dup


def main():
    try:                                                    # 控制台编码不支持的字符不要崩
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                       # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="粗筛合焦照片 (人像向): 解码 RAW + 眼睛锐度评分")
    ap.add_argument("folder", nargs="?", default=None,
                    help="照片目录; 不给就进入交互式, 直接把文件夹拖进窗口")
    ap.add_argument("--out", default=None, help="输出目录 (默认 <照片目录>/_cull_out)")
    ap.add_argument("--ext", default=None, help="扩展名, 逗号分隔 (默认 RAW, 自动跳过已有 RAW 的导出版)")
    ap.add_argument("--ratio", type=float, default=0.55, help="组内锐度低于最佳 x ratio 判为疑似模糊 (默认 0.55)")
    ap.add_argument("--group-ncc", type=float, default=0.85,
                    help="相似帧分组的指纹严格度 0~1 (默认 0.85; 这是实测能分开"
                         "'同一个人'和'换了人'的分界, 调到 0.7~0.8 会把"
                         "不同的人并进一组)")
    ap.add_argument("--group-window", type=float, default=BURST_SECS,
                    help=f"连拍分组的时间窗, 单位秒 (默认 {BURST_SECS:g}; 只有拍摄时间"
                         "相隔不超过这个值的帧才可能归为一组。设得越大越容易把"
                         "不同时段的照片并进来, 也越容易出现跨时段的大组)")
    ap.add_argument("--full", action="store_true", help="全分辨率解码 (更准, 慢约 5 倍)")
    ap.add_argument("--model", default=None, help="YuNet 模型路径")
    ap.add_argument("--jobs", type=int, default=4,
                    help="并行进程数 (默认 4; 实测解码受内存带宽限制, 4 个最快, 加到 8/16 反而更慢)")
    ap.add_argument("--sheet", action="store_true", help="额外导出检测结果总览图 (核对检测是否准确)")
    ap.add_argument("--preview-size", type=int, default=2048,
                    help="大图预览的最长边像素 (默认 2048; 0 = 不生成大图, 只能看内嵌小图)")
    ap.add_argument("--keep-to", default=None, help="把挑中的照片移动到这个文件夹")
    ap.add_argument("--keep-from", default=None,
                    help="从清单文件读取要移动的文件名 (网页上'导出勾选清单'的产物); 给这个就不再重新分析")
    ap.add_argument("--best-per-group", action="store_true",
                    help="每组只保留最锐的那张; 单张照片照原样保留 (只丢掉输给同伴的帧)")
    ap.add_argument("--min-ratio", type=float, default=0.0,
                    help="只保留组内占比 >= 这个比例的 (0~1; 只对多帧组有效, 单张视为通过)")
    ap.add_argument("--min-sharp", type=float, default=0.0, help="只保留眼部锐度 >= 这个值的")
    ap.add_argument("--copy", action="store_true", help="复制而不是移动 (更安全)")
    ap.add_argument("--with-jpg", action="store_true", help="连同名 .jpg 一起移动")
    ap.add_argument("--dry-run", action="store_true", help="只列出会移动哪些文件, 不动真格")
    ap.add_argument("--open", action="store_true", help="跑完自动用浏览器打开选图报告")
    ap.add_argument("--pause", action="store_true", help="跑完等按任意键 (双击/拖放启动时用)")
    args = ap.parse_args()

    target = args.folder
    if target is None:
        if not sys.stdin or not sys.stdin.isatty():
            target = "."
        else:
            print(BANNER)
            target = ask_folder()
            if not target:
                print("  已取消。")
                return
            args.open = True

    folder = os.path.abspath(target)
    if not os.path.isdir(folder):
        sys.exit(f"目录不存在: {folder}")
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(folder, "_cull_out")

    # ---- 只移动模式: 网页上导出清单后, 直接跑这一步, 不重新分析 ----
    if args.keep_from:
        if not os.path.isfile(args.keep_from):
            sys.exit(f"清单文件不存在: {args.keep_from}")
        if not args.keep_to:
            sys.exit("用 --keep-from 时必须同时给 --keep-to 指定目标文件夹")
        picked = parse_name_list(args.keep_from, tuple(RAW_EXTS + IMG_EXTS))
        print(f"从清单读到 {len(picked)} 个文件名: {args.keep_from}")
        do_move(folder, picked, args.keep_to, copy=args.copy, with_jpg=args.with_jpg,
                dry_run=args.dry_run, out_dir=out_dir)
        return

    if args.ext:
        wanted = tuple(e.strip().lower() for e in args.ext.split(","))
    else:
        wanted = tuple(RAW_EXTS + IMG_EXTS)
    names, n_all = list_photos(folder, wanted)
    if not names:
        sys.exit(f"在 {folder} 里没有可分析的文件")

    jobs = max(1, min(args.jobs, len(names)))
    model_path = ensure_model(args.model)
    os.makedirs(out_dir, exist_ok=True)
    preview_dir = None
    if args.preview_size > 0:
        preview_dir = os.path.join(out_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)

    print(f"待分析 {len(names)} 个文件"
          + (f" (跳过 {n_all - len(names)} 个已有 RAW 的 jpg 导出版)" if n_all > len(names) else ""))
    print(f"解码: {'全分辨率' if args.full else '半分辨率 (--full 更准)'}   "
          f"进程: {jobs}   阈值: 组内最佳 x {args.ratio}"
          + (f"   大图预览: {args.preview_size}px" if preview_dir else "   大图预览: 关") + "\n")

    rows = []
    cv_threads = 1 if jobs > 1 else 0
    if jobs > 1 and len(names) > 1:
        with HardStopPool(max_workers=jobs, initializer=worker_init,
                          initargs=(model_path, args.full, preview_dir,
                                    args.preview_size, cv_threads)) as ex:
            futs = [ex.submit(process_one, n, folder) for n in names]
            try:
                for i, f in enumerate(as_completed(futs), 1):
                    rec = f.result()
                    rows.append(rec)
                    print_progress(i, len(names), rec)
            except KeyboardInterrupt:
                # Ctrl+C: ex.__exit__ 会 terminate 掉所有 worker, 不然它们还会
                # 接着把这批照片解码完才肯退出。
                ex.stop()
                raise
    else:
        worker_init(model_path, args.full, preview_dir, args.preview_size, cv_threads)
        for i, n in enumerate(names, 1):
            rec = process_one(n, folder)
            rows.append(rec)
            print_progress(i, len(names), rec)

    order = {n: i for i, n in enumerate(names)}             # 多进程完成顺序是乱的, 排回去
    rows.sort(key=lambda r: order.get(r["file"], 0))

    # ---- 分组 + 报告 ----
    groups, n_soft, n_face, n_dup = write_reports(rows, folder, out_dir,
                                                  full=args.full, groups_ncc=args.group_ncc,
                                                  soft_ratio=args.ratio,
                                                  group_window=args.group_window)
    print(f"\n{len(rows)} 张 / {len(groups)} 组, 检出人脸 {n_face} 张, "
          f"可对比的相似帧 {n_dup} 组, 其中疑似模糊 {n_soft} 张")
    multi = sorted([g for g in groups if len(g) > 1], key=lambda g: -len(g))
    if multi:
        print(f"\n几乎同一姿势的 {len(multi)} 组 (只有这些组内比较才可靠):")
        for g in multi[:20]:
            g = sorted(g, key=lambda r: -r["compare_value"])
            print(f'  组{g[0]["group"]:<3}({len(g)}张) 保留 {g[0]["file"]:<15}'
                  f'  |  其余: ' + ", ".join(f'{r["file"]}({r["ratio_group"]:.0%})' for r in g[1:]))
    else:
        print("\n这批照片里没有找到几乎同一姿势的帧 (都是单张), 因此没有做自动模糊判定。")
    if n_soft:
        print(f"\n组内疑似模糊 {n_soft} 张:")
        for r in sorted(rows, key=lambda x: x["compare_value"]):
            if r.get("soft"):
                print(f'  {r["file"]:<16} 眼部锐度 {r["eye_sharp"]:>8}  '
                      f'组{r.get("group")} 占最佳 {r.get("ratio_group",0):.0%}')
    top = [r for r in sorted(rows, key=lambda x: -x.get("compare_value", 0)) if r.get("subject") == "face"]
    print(f"\n眼部锐度最高的 10 张 (可优先看):")
    for r in top[:10]:
        print(f'  {r["file"]:<16} {r["eye_sharp"]:>8}   脸宽 {r.get("face_w",0):>4}px  '
              f'{fmt_rat(r.get("focal_length"),0)}mm')

    if args.sheet:
        print("\n检测总览图: " + make_sheet(rows, os.path.join(out_dir, "检测总览.jpg")))
    # ---- 按条件挑片并移动 ----
    if args.keep_to:
        picked = select_rows(rows, best_per_group=args.best_per_group,
                             min_ratio=args.min_ratio, min_sharp=args.min_sharp)
        cond = []
        if args.best_per_group:
            cond.append("每组只留最锐的")
        if args.min_ratio > 0:
            cond.append(f"组内占比>={args.min_ratio:.0%}")
        if args.min_sharp > 0:
            cond.append(f"锐度>={args.min_sharp:g}")
        n_single = sum(1 for r in picked if r.get("group_size", 1) <= 1)
        print(f"\n筛选条件: {', '.join(cond) if cond else '无(全部)'}  ->  命中 {len(picked)}/{len(rows)} 张"
              f"  (其中单张 {n_single} 张, 组内胜出 {len(picked) - n_single} 张)")
        print("  被排除的是输给同组同伴的帧; 单张照片不会被排除。")
        do_move(folder, [r["file"] for r in picked], args.keep_to, copy=args.copy,
                with_jpg=args.with_jpg, dry_run=args.dry_run, out_dir=out_dir)
        if not args.dry_run:
            with open(os.path.join(out_dir, "保留清单.txt"), "w", encoding="utf-8") as fh:
                for r in picked:
                    fh.write(f'{r["file"]}\n')

    if args.open:
        html_path = os.path.join(out_dir, "选图报告.html")
        try:
            os.startfile(html_path)                          # Windows: 用默认浏览器打开
            print("\n已打开选图报告。")
        except Exception:                                    # noqa: BLE001
            import webbrowser
            webbrowser.open("file:///" + html_path.replace("\\", "/").replace(" ", "%20"))


def wait_key():
    """等一次按键, 让双击启动的窗口不要立刻关掉。"""
    print("\n按任意键关闭窗口 ...", end="", flush=True)
    try:
        import msvcrt
        msvcrt.getch()
        print()
    except Exception:                                       # noqa: BLE001
        try:
            input()
        except Exception:                                   # noqa: BLE001
            pass


def make_sheet(rows, out_path, cols=6, tw=320):
    imgs = []
    for r in rows:
        t = r.get("_thumb")
        imgs.append(Image.open(io.BytesIO(base64.b64decode(t.split(",", 1)[1]))) if t else None)
    rows_n = (len(rows) + cols - 1) // cols
    ch = tw + 18
    sheet = Image.new("RGB", (cols * tw, rows_n * ch), (20, 20, 24))
    d = ImageDraw.Draw(sheet)
    for i, (r, im) in enumerate(zip(rows, imgs)):
        x, y = (i % cols) * tw, (i // cols) * ch
        if im is None:
            continue
        sheet.paste(im.resize((tw, round(im.height * tw / im.width))), (x, y))
        d.text((x + 3, y + tw + 2),
               f'{r["file"][:-4]} 眼{r["eye_sharp"]:.0f} {r.get("face_w",0)}px', fill=(235, 235, 235))
    sheet.save(out_path, quality=80)


if __name__ == "__main__":
    # 打包成 exe 后, 多进程 worker 会重新导入本模块, 需要这个防止递归
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:                                       # noqa: BLE001
        pass
    try:
        main()
    finally:
        if "--pause" in sys.argv:
            wait_key()
