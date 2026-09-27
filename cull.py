#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粗筛合焦照片 —— 人像/活动向的对焦预筛

思路:
  大光圈人像的清晰度必须看「眼睛」, 不能看整幅画面。
  1. rawpy(libraw) 真正解码 RAW, 拿到原始像素
  2. OpenCV YuNet(轻量 CNN 人脸检测) 找脸, 并给出眼睛等 5 个关键点
  3. 在眼睛位置的原始像素上算 Tenengrad 梯度能量 —— 合焦的眼睛边缘锐利,
     跑焦的边缘被抹平, 这个指标会明显掉下来
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
import html
import io
import json
import math
import os
import pathlib
import shutil
import struct
import sys
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
# 模型 / 中文路径
# ==========================================================================
def _ascii_dir():
    """OpenCV 在 Windows 下打不开含中文路径的文件, 模型要放到纯英文目录。"""
    import tempfile
    for cand in (os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "cmdc_cull"),
                 os.path.join(tempfile.gettempdir(), "cmdc_cull")):
        try:
            cand.encode("ascii")
            os.makedirs(cand, exist_ok=True)
            return cand
        except (UnicodeEncodeError, OSError):
            continue
    return None


def ensure_model(path=None):
    if path and os.path.isfile(path):
        return path
    d = _ascii_dir()
    if not d:
        sys.exit("找不到可写的纯英文目录, 请用 --model 指定 YuNet 模型路径")
    dst = os.path.join(d, YUNET_NAME)
    if not os.path.isfile(dst) or os.path.getsize(dst) < 100_000:
        print(f"首次运行: 下载人脸检测模型 ({YUNET_NAME}) ...")
        try:
            urllib.request.urlretrieve(YUNET_URL, dst)
        except Exception as exc:                             # noqa: BLE001
            sys.exit(f"模型下载失败: {exc}\n可以手动下载后放到 {dst} 或用 --model 指定位置")
    return dst


def make_detector(model_path, score_thr=0.6):
    return cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                     score_threshold=score_thr,
                                     nms_threshold=0.3, top_k=5000)


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


def worker_init(model_path, full, preview_dir=None, preview_side=2048, cv_threads=1):
    # 多进程时要把 OpenCV 自己的线程数压下来: 默认它会开满所有核,
    # 4 个进程 x 20 线程 = 80 线程抢 20 个核, 反而把整体拖慢。
    if cv_threads and cv_threads > 0:
        cv2.setNumThreads(cv_threads)
    _W["det"] = make_detector(model_path)
    _W["full"] = full
    _W["preview_dir"] = preview_dir
    _W["preview_side"] = preview_side


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
                    "_eye_boxes": [], "sig": None,
                    "face_w": 0, "face_cx": 0, "face_cy": 0, "face_area_pct": 0,
                    "face_conf": 0, "brightness": 0, "clipped": 0, "decoded": "", "_ms": 0})
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
    used_boxes = []
    if len(faces):
        f = max(faces, key=lambda v: v[2] * v[3])
        fx, fy, fw, fh = [v * inv for v in f[:4]]
        lms = (f[4:14].reshape(5, 2) * inv)
        conf = float(f[-1])

        # 眼睛区域锐度 (原图分辨率上算)
        er = max(10, fw * 0.24)
        eyes = []
        for ex, ey in (lms[0], lms[1]):
            t, c = patch_sharpness(gray, ex, ey, er)
            if t > 0:
                eyes.append(t)
        eye_sharp = float(np.mean(eyes)) if eyes else 0.0
        face_ten, face_std = patch_sharpness(gray, fx + fw / 2, fy + fh / 2, max(fw, fh) * 0.55)

        rec.update({
            "eye_sharp": round(eye_sharp, 1),
            "face_sharp": round(face_ten, 1),
            "face_conf": round(conf, 2),
            "face_w": int(fw),
            "face_cx": round((fx + fw / 2) / W, 4),
            "face_cy": round((fy + fh / 2) / H, 4),
            "face_area_pct": round(fw * fh / (W * H) * 100, 2),
            "sig": face_signature(gray, (fx, fy, fw, fh)),
            "_face_box": [int(fx), int(fy), int(fw), int(fh)],
            "_eye_boxes": [[int(ex - er), int(ey - er), int(2 * er), int(2 * er)] for ex, ey in (lms[0], lms[1])],
        })
        subject_label = "face"
        used_boxes = rec["_face_box"][:]
        rec["compare_value"] = rec["eye_sharp"]
    else:
        t, c = patch_sharpness(gray, W * 0.5, H * 0.45, min(W, H) * 0.22)
        rec.update({"eye_sharp": 0.0, "face_sharp": round(t, 1), "face_conf": 0.0,
                    "face_w": 0, "face_cx": 0.5, "face_cy": 0.45, "face_area_pct": 0,
                    "sig": None, "_face_box": None, "_eye_boxes": []})
        rec["compare_value"] = rec["face_sharp"]

    rec["subject"] = subject_label
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    rec["sharp_global"] = round(float(tenengrad_map(g).mean()), 1)
    rec["brightness"] = round(float(gray.mean()), 1)
    rec["clipped"] = round(float(((gray > 250).sum() + (gray < 5).sum()) / gray.size * 100), 2)
    rec["_thumb"] = make_thumb(bgr, rec["_face_box"], rec["_eye_boxes"], af_point)
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


def make_thumb(bgr, face_box, eye_boxes, af_point=None, max_w=360, quality=78):
    """用 cv2 缩略图 + 画框 (比 PIL LANCZOS 快数倍, 实测 84ms -> ~15ms)。"""
    h, w = bgr.shape[:2]
    k = max_w / w
    im = cv2.resize(bgr, (max_w, max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    if face_box:
        x, y, bw, bh = [int(round(v * k)) for v in face_box]
        cv2.rectangle(im, (x, y), (x + bw, y + bh), (64, 96, 255), 2)        # 红 = 人脸
    for x, y, bw, bh in (eye_boxes or []):
        x, y, bw, bh = [int(round(v * k)) for v in (x, y, bw, bh)]
        cv2.rectangle(im, (x, y), (x + bw, y + bh), (255, 210, 0), 1)        # 青 = 眼睛区域
    if af_point:                                                            # 绿 = 相机设定的对焦点
        ax, ay = int(round(af_point[0] * im.shape[1])), int(round(af_point[1] * im.shape[0]))
        cv2.circle(im, (ax, ay), 13, (0, 220, 0), 2)
        cv2.drawMarker(im, (ax, ay), (0, 220, 0), cv2.MARKER_CROSS, 17, 2)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ==========================================================================
# 分组: 同姿势 / 同机位才算可比
# ==========================================================================
def ncc(a, b):
    return float(np.dot(a, b) / a.size)


def group_frames(rows, pos_tol=0.035, size_tol=0.12, ncc_min=0.80, focal_tol=0.06):
    """只在「几乎同一姿势」的帧之间做对比。

    阈值故意收得很紧: 经验上整组人像里只有真正连拍/复拍的那几对能进来,
    松一点就会把不同姿势的帧并在一起, 于是"组内最佳"和"疑似模糊"都会误报。
    """
    groups = []
    for r in sorted(rows, key=lambda x: x["file"]):
        if r.get("subject") != "face" or r.get("sig") is None:
            groups.append([r])
            continue
        placed = False
        for g in groups:
            rep = g[0]
            if rep.get("subject") != "face" or rep.get("sig") is None:
                continue
            if abs(math.log2(max(r["face_w"], 1) / max(rep["face_w"], 1))) > size_tol:
                continue
            if math.hypot(r["face_cx"] - rep["face_cx"], r["face_cy"] - rep["face_cy"]) > pos_tol:
                continue
            fr, fp = r.get("focal_length"), rep.get("focal_length")
            if fr and fp and abs(math.log2((fr[0] / fr[1]) / (fp[0] / fp[1]))) > focal_tol:
                continue
            if ncc(r["sig"], rep["sig"]) < ncc_min:
                continue
            g.append(r)
            placed = True
            break
        if not placed:
            groups.append([r])
    return groups


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
def compute_flags(rows, groups_ncc=0.80, soft_ratio=0.55):
    """给 rows 打上分组/判定标记。纯计算, 不碰磁盘。

    GUI 直接调这个函数出结果, 不经过 write_reports —— 所以 exe 里不用写任何文件。
    """
    groups = group_frames([r for r in rows if not r.get("_error")], ncc_min=groups_ncc)
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
    for r in rows:
        r.setdefault("group", 0)
        r.setdefault("group_size", 1)
        r.setdefault("ratio_group", 0)
        r.setdefault("best_in_group", False)
        r.setdefault("soft", bool(r.get("_error")))

    return {
        "groups": groups,
        "n_soft": sum(1 for r in rows if r.get("soft")),
        "n_face": sum(1 for r in rows if r.get("faces")),
        "n_dup": sum(1 for g in groups if len(g) > 1),
    }


CSV_COLS = [("group", "组"), ("group_size", "组内张数"), ("best_in_group", "组内最佳"),
            ("soft", "疑似模糊"), ("file", "文件名"), ("eye_sharp", "眼部锐度"),
            ("ratio_group", "占组内最佳"), ("face_sharp", "脸部锐度"),
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


def write_reports(rows, folder, out_dir, full=False, groups_ncc=0.80, soft_ratio=0.55):
    """给 rows 打上分组/判定标记, 并写出 CSV / txt / HTML。返回统计数字。"""
    st = compute_flags(rows, groups_ncc=groups_ncc, soft_ratio=soft_ratio)
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
        fh.write("(顺序 = 眼部锐度从高到低; 数值受框景/光线影响, 不要跨照片直接比)\n\n")
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
              '<b>重要</b>: 锐度是眼睛区域的梯度能量, 会受框景大小和光线影响, '
              '<b>只有同一组的帧之间比较才有意义</b>, 不要拿绝对值跨照片比。'
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
    ap.add_argument("--group-ncc", type=float, default=0.80,
                    help="相似帧分组的严格度 0~1 (默认 0.80, 越大越严; 0.7 更宽松但可能误判)")
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
        with ProcessPoolExecutor(max_workers=jobs, initializer=worker_init,
                                 initargs=(model_path, args.full, preview_dir,
                                           args.preview_size, cv_threads)) as ex:
            futs = [ex.submit(process_one, n, folder) for n in names]
            for i, f in enumerate(as_completed(futs), 1):
                rec = f.result()
                rows.append(rec)
                print_progress(i, len(names), rec)
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
                                                  soft_ratio=args.ratio)
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
