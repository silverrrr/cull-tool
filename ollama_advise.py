#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 Ollama 调色建议模块 (独立、无项目内依赖)

职责分两块:
  1. compute_color_stats(): 从解码后的 BGR 图像算客观色彩统计
     (RGB 均值/分位、高光溢出、死黑、对比度、饱和度、色度、R/B 比、曝光估计)
  2. check_ollama() / analyze_images(): 连本机 Ollama (默认 127.0.0.1:11434),
     把一组照片的缩略 JPEG + 客观统计喂给视觉模型, 拿回整组风格判断和逐张微调值

设计要点:
  - 只用 stdlib + cv2 + numpy, 不 import 本项目任何模块 (避免循环依赖)
  - 所有对外数值一律转成 python 原生 float/int —— numpy 标量 json.dumps 会直接报错
  - 优先用 Ollama >=0.5.0 的结构化输出 format=<JSON Schema>; 老版本不认 schema 会回
    HTTP 400, 此时退回 format="json" 再试一次
  - images 用裸 base64 (不带 data:image/... 前缀), 这是 Ollama /api/chat 的约定
"""

import base64
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

# OLLAMA_URL 仅作为文档化的默认值 / 向后兼容常量保留; 运行时一律走 base_url()
# (base_url() 优先返回 set_base_url() 的运行时覆盖, 否则读 cull_config.json)
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_MAX_IMAGES = 24
DEFAULT_TIMEOUT = 300.0

# 流式请求时 on_progress 的最小回调间隔(秒): 首次立即回调, 之后最多这么频繁
_PROGRESS_INTERVAL_S = 0.4

# 老版本 Ollama 的 /api/show 不返回 capabilities, 只能靠模型名里是否含这些关键词猜视觉能力
VISION_HINTS = (
    "llava", "bakllava", "qwen2.5vl", "qwen2-vl", "qwen3-vl",
    "llama3.2-vision", "minicpm-v", "moondream", "granite3.2-vision", "gemma3",
)
# gemma3 只有 4b/12b/27b 带视觉, 270m/1b 是纯文本
_TEXT_ONLY_GEMMA3 = ("270m", "1b")

# 逐张建议的数值字段 (缺失一律补 0.0)
_NUM_KEYS = (
    "exposure", "temperature", "contrast", "highlights",
    "shadows", "saturation", "vibrance",
)

# Ollama 结构化输出的 JSON Schema, 形状必须和最终返回的 advice dict 一致
ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "group_style": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "exposure": {"type": "number"},
                    "temperature": {"type": "number"},
                    "contrast": {"type": "number"},
                    "highlights": {"type": "number"},
                    "shadows": {"type": "number"},
                    "saturation": {"type": "number"},
                    "vibrance": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["name", *_NUM_KEYS, "reason"],
            },
        },
    },
    "required": ["group_style", "issues", "suggestions", "images"],
}


# ---------------------------------------------------------------- 连接设置
#
# 设置持久化在工具目录的 cull_config.json 里, 形状 (只占用 "ollama" 块,
# 不影响 cull.py 的 "weights" / "model_dir" 等顶层键):
#
#   {
#     "weights": { ... },              # cull.py 的
#     "ollama": {
#       "host": "127.0.0.1",           # 必填, 非空
#       "port": 11434,                 # 1~65535
#       "scheme": "http",              # http 或 https
#       "model": "qwen2.5vl:7b",       # 默认调色建议模型
#       "max_images": 24,              # 单次最多发多少张
#       "timeout": 300.0               # 单次请求超时(秒)
#     }
#   }
#
# 配置文件路径与 cull.load_config 保持一致: 先在 _MEIPASS/工具目录的上级找,
# 再在工具目录里找。打包运行时绝不写进 sys._MEIPASS (那是临时解压目录),
# 而是写在 exe 同级。

# 默认 "ollama" 块 (可被 cull_config.json 覆盖)
_DEFAULT_OLLAMA = {
    "host": "127.0.0.1",
    "port": 11434,
    "scheme": "http",
    "model": DEFAULT_MODEL,
    "max_images": DEFAULT_MAX_IMAGES,
    "timeout": DEFAULT_TIMEOUT,
}

_CFG_CACHE = None      # 解析后的 cull_config.json 原始 dict; 惰性加载
_BASE_OVERRIDE = None  # set_base_url() 设置的运行时地址覆盖


def _base_dir():
    """工具根目录: 打包后 = sys._MEIPASS; 源码 = 本文件所在目录。"""
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))


def _config_candidates():
    """按 cull.load_config 的顺序给出候选 cull_config.json 路径。"""
    base = _base_dir()
    return [os.path.join(base, os.pardir, "cull_config.json"),
            os.path.join(base, "cull_config.json")]


def _read_config_path():
    """读取用的配置路径: 第一个存在的候选; 都不存在返回 None。"""
    for cand in _config_candidates():
        if os.path.isfile(cand):
            return cand
    return None


def _write_config_path():
    """写入用的配置路径: 优先已有的那个; 否则源码写工具目录, 打包写 exe 同级。"""
    found = _read_config_path()
    if found:
        return found
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "cull_config.json")
    return os.path.join(_base_dir(), "cull_config.json")


def _invalidate_config():
    """清掉配置缓存, 让下一次读取重新解析 (save_settings 后调用)。"""
    global _CFG_CACHE
    _CFG_CACHE = None


def _load_raw_config():
    """读 cull_config.json 的原始 dict, 缓存到模块全局。损坏/缺失一律返回 {}。"""
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    cfg = {}
    path = _read_config_path()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                cfg = data
        except Exception:                    # noqa: BLE001  损坏配置不能让它炸
            cfg = {}
    _CFG_CACHE = cfg
    return cfg


def _as_int(value):
    """尽力转 int; 布尔或不可转返回 None。"""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    """尽力转 float; 布尔或不可转返回 None。"""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_url(text):
    """把 "127.0.0.1:11434" / "http://192.168.1.9:11434" 解析成 {scheme, host, port}。

    缺协议补 http, 缺端口补 11434。非法或非 http(s) 返回 None。
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parts = urllib.parse.urlparse(raw)
    except ValueError:
        return None
    scheme = (parts.scheme or "http").strip().lower()
    if scheme not in ("http", "https"):
        return None
    host = parts.hostname
    if not host:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port is None:
        port = _DEFAULT_OLLAMA["port"]
    return {"scheme": scheme, "host": host, "port": port}


def get_settings():
    """读当前 Ollama 设置 (配置缺项/非法一律回落到默认值)。

    返回:
        {"host": str, "port": int, "scheme": str, "model": str,
         "max_images": int, "timeout": float, "url": str, "config_path": str}

    其中 url = f"{scheme}://{host}:{port}"; config_path 是"若现在保存会写到哪"。
    配置文件里 "ollama" 块的形状见本段顶部注释。
    """
    raw = _load_raw_config()
    block = raw.get("ollama") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        block = {}

    host = block.get("host")
    host = str(host).strip() if isinstance(host, str) else ""
    if not host:
        host = _DEFAULT_OLLAMA["host"]

    port = _as_int(block.get("port"))
    if port is None or not (1 <= port <= 65535):
        port = _DEFAULT_OLLAMA["port"]

    scheme = block.get("scheme")
    scheme = str(scheme).strip().lower() if isinstance(scheme, str) else ""
    if scheme not in ("http", "https"):
        scheme = _DEFAULT_OLLAMA["scheme"]

    model = block.get("model")
    model = str(model).strip() if isinstance(model, str) else ""
    if not model:
        model = DEFAULT_MODEL

    max_n = _as_int(block.get("max_images"))
    if max_n is None or max_n < 1:
        max_n = DEFAULT_MAX_IMAGES

    timeout = _as_float(block.get("timeout"))
    if timeout is None or timeout <= 0:
        timeout = DEFAULT_TIMEOUT

    return {
        "host": host,
        "port": port,
        "scheme": scheme,
        "model": model,
        "max_images": max_n,
        "timeout": timeout,
        "url": f"{scheme}://{host}:{port}",
        "config_path": _write_config_path(),
    }


def save_settings(settings):
    """保存 Ollama 连接设置到 cull_config.json 的 "ollama" 块, 并清掉运行时覆盖。

    settings 可含 host / port / scheme / model / max_images / timeout, 另支持
    便捷键 url (如 "127.0.0.1:11434" 或 "http://192.168.1.9:11434"): 给了 url
    就自动解析出 scheme/host/port (缺协议补 http, 缺端口补 11434)。
    只覆盖 "ollama" 块, 保留 weights / model_dir 等其它顶层键。
    JSON 以 UTF-8 + indent=2 + ensure_ascii=False 写入。
    返回 {"ok": True, "settings": <get_settings()>, "path": <写入路径>}
    或 {"ok": False, "error": <中文>}。
    """
    if not isinstance(settings, dict):
        return {"ok": False, "error": "设置必须是字典"}

    cur = get_settings()
    merged = {
        "host": cur["host"],
        "port": cur["port"],
        "scheme": cur["scheme"],
        "model": cur["model"],
        "max_images": cur["max_images"],
        "timeout": cur["timeout"],
    }

    url = settings.get("url")
    if url is not None and str(url).strip():
        parsed = _parse_url(url)
        if parsed is None:
            return {"ok": False, "error": "URL 格式不合法"}
        merged.update(parsed)

    if "host" in settings:
        host = str(settings.get("host") or "").strip()
        if not host:
            return {"ok": False, "error": "主机地址不能为空"}
        merged["host"] = host

    if "port" in settings:
        port = _as_int(settings.get("port"))
        if port is None or not (1 <= port <= 65535):
            return {"ok": False, "error": "端口必须是 1~65535 的整数"}
        merged["port"] = port

    if "scheme" in settings:
        scheme = str(settings.get("scheme") or "").strip().lower()
        if scheme not in ("http", "https"):
            return {"ok": False, "error": "协议只能是 http 或 https"}
        merged["scheme"] = scheme

    if "model" in settings:
        model = str(settings.get("model") or "").strip()
        if not model:
            return {"ok": False, "error": "模型名不能为空"}
        merged["model"] = model

    if "max_images" in settings:
        max_n = _as_int(settings.get("max_images"))
        if max_n is None or max_n < 1:
            return {"ok": False, "error": "最大图片数必须是 ≥1 的整数"}
        merged["max_images"] = max_n

    if "timeout" in settings:
        timeout = _as_float(settings.get("timeout"))
        if timeout is None or timeout <= 0:
            return {"ok": False, "error": "超时必须是正数"}
        merged["timeout"] = timeout

    path = _write_config_path()
    raw = _load_raw_config()
    data = dict(raw) if isinstance(raw, dict) else {}
    data["ollama"] = merged
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        return {"ok": False, "error": f"写入配置失败: {exc}"}

    _invalidate_config()
    global _BASE_OVERRIDE
    _BASE_OVERRIDE = None
    return {"ok": True, "settings": get_settings(), "path": path}


def base_url():
    """当前生效的 Ollama 根地址: 运行时覆盖优先, 否则读配置里的 url。"""
    if _BASE_OVERRIDE:
        return _BASE_OVERRIDE
    return get_settings()["url"]


def set_base_url(url):
    """设置运行时覆盖地址; 传空串/None 清除覆盖, 回落到配置。

    地址没带协议时自动补 http://, 末尾斜杠会被去掉。
    """
    global _BASE_OVERRIDE
    text = str(url or "").strip()
    if not text:
        _BASE_OVERRIDE = None
        return
    if "://" not in text:
        text = "http://" + text
    _BASE_OVERRIDE = text.rstrip("/")


def default_model():
    """当前默认调色建议模型名。"""
    return get_settings()["model"]


def max_images():
    """单次调色建议最多发送的图片数。"""
    return get_settings()["max_images"]


def request_timeout():
    """单次请求 Ollama 的超时秒数。"""
    return get_settings()["timeout"]


def _py(value):
    """把 numpy 标量/数组转成 python 原生对象。numpy 标量不能 json.dumps。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _num(value, default=0.0):
    """尽力转成 float, 失败或不可转时返回 default (供模型输出兜底)。"""
    try:
        return float(_py(value))
    except (TypeError, ValueError):
        return float(default)


def _empty_stats():
    z = [0.0, 0.0, 0.0]
    return {
        "mean_rgb": list(z), "p1_rgb": list(z), "p50_rgb": list(z), "p99_rgb": list(z),
        "clip_hi_pct": 0.0, "clip_lo_pct": 0.0, "contrast_std": 0.0,
        "sat_mean": 0.0, "chroma_mean": 0.0, "wb_rb_ratio": 0.0,
        "exposure_ev_est": 0.0,
    }


def compute_color_stats(bgr):
    """从 BGR (HxWx3 uint8) 算客观色彩统计, 返回值全是 python 原生类型。"""
    if bgr is None or getattr(bgr, "size", 0) == 0 or bgr.ndim != 3:
        return _empty_stats()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    def _mean_rgb():
        return [float(r.mean()), float(g.mean()), float(b.mean())]

    def _pct_rgb(p):
        return [float(np.percentile(r, p)), float(np.percentile(g, p)),
                float(np.percentile(b, p))]

    mean_rgb = _mean_rgb()
    # 高光溢出 = 三通道最大值触顶; 死黑 = 三通道最小值触底
    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    clip_hi = float((max_c >= 250).mean() * 100.0)
    clip_lo = float((min_c <= 5).mean() * 100.0)

    contrast_std = float(gray.std())
    sat_mean = float(hsv[:, :, 1].mean())

    # LAB 的 a/b 以 128 为中性; 减掉 128 再算半径才是真正的色度
    a = lab[:, :, 1].astype(np.float32) - 128.0
    bb = lab[:, :, 2].astype(np.float32) - 128.0
    chroma_mean = float(np.sqrt(a * a + bb * bb).mean())

    r_mean, b_mean = mean_rgb[0], mean_rgb[2]
    wb_rb_ratio = float(r_mean / max(b_mean, 1.0))

    median_gray = float(np.median(gray))
    ev = math.log2(118.0 / max(median_gray, 1.0))
    ev = round(max(-3.0, min(3.0, ev)), 2)

    return {
        "mean_rgb": mean_rgb,
        "p1_rgb": _pct_rgb(1),
        "p50_rgb": _pct_rgb(50),
        "p99_rgb": _pct_rgb(99),
        "clip_hi_pct": clip_hi,
        "clip_lo_pct": clip_lo,
        "contrast_std": contrast_std,
        "sat_mean": sat_mean,
        "chroma_mean": chroma_mean,
        "wb_rb_ratio": wb_rb_ratio,
        "exposure_ev_est": ev,
    }


def jpeg_bytes(bgr, side=1024):
    """把图缩到长边 side 并编码成 JPEG(q85) 字节。只缩不放, 小图原样编码。"""
    img = bgr
    h, w = img.shape[:2]
    longest = max(h, w)
    if side and side > 0 and longest > side:
        scale = float(side) / float(longest)
        img = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    img = np.ascontiguousarray(img)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


# ---------------------------------------------------------------- HTTP 辅助

def _get_json(path, timeout):
    req = urllib.request.Request(base_url() + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(path, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url() + path, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _norm_model(name):
    """把 x 和 x:latest 视作同一个模型。"""
    n = str(name or "").strip()
    return n if ":" in n else (n + ":latest" if n else "")


def _read_error_body(exc):
    try:
        text = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001
        text = ""
    return text or str(getattr(exc, "reason", "") or "")


# ---------------------------------------------------------------- 健康检查

def check_ollama(model=None, timeout=3.0):
    """探测本机 Ollama: 是否在跑、目标模型是否已 pull、是否真的落在 GPU 上。

    model 为空时用配置里的默认模型 (default_model())。
    """
    model = model or default_model()
    connect_hint = "无法连接本地 Ollama (127.0.0.1:11434), 请先启动 ollama serve"
    try:
        _get_json("/api/version", timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
        return {"ok": False, "need_pull": False, "error": connect_hint}

    try:
        tags = _get_json("/api/tags", timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
        return {"ok": False, "need_pull": False, "error": connect_hint}

    want = _norm_model(model)
    models = tags.get("models") if isinstance(tags, dict) else None
    present = any(
        _norm_model(m.get("name")) == want or _norm_model(m.get("model")) == want
        for m in (models or []) if isinstance(m, dict)
    )
    if not present:
        return {
            "ok": False, "need_pull": True, "model": model,
            "error": f"模型 {model} 尚未 pull。请先运行: ollama pull {model}",
        }

    # /api/ps 只作诊断: size_vram 为 0/缺失说明这次是 CPU 推理 (不致命)
    gpu = None
    try:
        ps = _get_json("/api/ps", timeout)
        loaded = ps.get("models") if isinstance(ps, dict) else None
        for m in (loaded or []):
            if not isinstance(m, dict):
                continue
            if _norm_model(m.get("name")) == want or _norm_model(m.get("model")) == want:
                gpu = bool(m.get("size_vram"))
                break
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
        gpu = None

    return {"ok": True, "model": model, "gpu": gpu, "model_present": True}


# ---------------------------------------------------------------- 模型管理

def _show_capabilities(name, timeout):
    """POST /api/show 取该模型的 capabilities 列表。

    新服务器认 {"model": ...}, 老服务器认 {"name": ...}; 前者返回 HTTP 400 时
    换键重试一次。任何失败都返回 None —— 单个模型探测失败不能中断整轮列举,
    调用方会退回名字启发式。
    """
    data = None
    for key in ("model", "name"):
        try:
            data = _post_json("/api/show", {key: name}, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and key == "model":
                continue  # 老服务器只认 name, 换键重试一次
            return None
        except (urllib.error.URLError, OSError, ValueError):
            return None
        break

    if not isinstance(data, dict):
        return None
    caps = data.get("capabilities")
    if not caps:
        return None  # 老服务器没有该字段 -> 退回名字启发式
    return [str(c) for c in caps]


def _gemma3_is_text_only(name, parameter_size):
    """gemma3 的 270m/1b 是纯文本模型, 名字里带 gemma3 也不能当视觉模型。"""
    low = str(name or "").lower()
    tag = low.split(":", 1)[1].strip() if ":" in low else ""
    size = str(parameter_size or "").strip().lower()
    return any(tok == tag for tok in _TEXT_ONLY_GEMMA3) or any(
        tok == size for tok in _TEXT_ONLY_GEMMA3
    )


def _vision_by_name(name, parameter_size):
    """老版本 Ollama 的兜底: 按模型名是否含已知视觉关键词判断。"""
    low = str(name or "").lower()
    if not any(hint in low for hint in VISION_HINTS):
        return False
    return not _gemma3_is_text_only(name, parameter_size)


def _is_vision_model(name, parameter_size, caps):
    """有 capabilities 就只认它; 缺/空才退回名字启发式 (老 Ollama)。"""
    if caps is not None:
        return "vision" in caps
    return _vision_by_name(name, parameter_size)


def list_models(timeout=3.0):
    """列出本机已 pull 的视觉模型。

    返回 {"ok": True, "models": [{"name", "size", "parameter_size", "vision"}, ...]}
    或 {"ok": False, "error": <中文>} (连不上 Ollama)。只保留支持视觉的模型, 按名字排序。
    """
    connect_hint = "无法连接本地 Ollama (127.0.0.1:11434), 请先启动 ollama serve"
    try:
        tags = _get_json("/api/tags", timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
        return {"ok": False, "error": connect_hint}

    raw_models = tags.get("models") if isinstance(tags, dict) else None
    out = []
    for m in (raw_models or []):
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        details = m.get("details")
        parameter_size = ""
        if isinstance(details, dict):
            parameter_size = str(details.get("parameter_size") or "")
        if not _is_vision_model(name, parameter_size, _show_capabilities(name, timeout)):
            continue
        try:
            size = int(m.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        out.append({
            "name": name,
            "size": size,
            "parameter_size": parameter_size,
            "vision": True,
        })

    out.sort(key=lambda item: item["name"])
    return {"ok": True, "models": out}


def _maybe_int(value):
    """把 JSON 里的 total/completed 规整成 python int, 缺失/非法返回 None。"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def pull_model(model, on_progress=None, cancel=None, timeout=60.0):
    """流式拉取(下载)一个模型, 成功返回 {"ok": True, "model": model}。

    逐行读 /api/pull 的 NDJSON 响应, 每个非空行解析后调用 on_progress
    ({"status", "total", "completed"}); cancel 是 threading.Event, 置位即中止。
    失败/取消一律抛带中文说明的 RuntimeError。timeout 是单次读取超时, 不是总时长。
    """
    name = str(model or "").strip()
    if not name:
        raise RuntimeError("模型名不能为空")

    body = json.dumps({"model": name, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        base_url() + "/api/pull", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        if exc.code == 404:
            raise RuntimeError(f"拉取失败: Ollama 找不到模型 {name} (HTTP 404) {detail}".strip()) from exc
        raise RuntimeError(f"拉取失败 (HTTP {exc.code}): {detail}".strip()) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            "无法连接本地 Ollama (127.0.0.1:11434), 请先启动 ollama serve"
        ) from exc

    with resp:
        for raw in resp:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("已取消")
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            err = obj.get("error")
            if err:
                raise RuntimeError(f"拉取失败: {err}")
            if callable(on_progress):
                on_progress({
                    "status": str(obj.get("status") or ""),
                    "total": _maybe_int(obj.get("total")),
                    "completed": _maybe_int(obj.get("completed")),
                })

    return {"ok": True, "model": name}


# ---------------------------------------------------------------- 建议调用

def _stats_table(stats):
    """把客观统计压成紧凑文本表, 让模型基于真实数字而非猜测来推理。"""
    def _vec(v):
        try:
            return "/".join(f"{float(x):.0f}" for x in v)
        except (TypeError, ValueError):
            return "-"

    lines = ["name | RGB均值 | P1 | P50 | P99 | 高光溢出% | 死黑% | 对比std | 饱和 | 色度 | R/B | 曝光EV"]
    for s in (stats or []):
        if not isinstance(s, dict):
            continue
        lines.append(" | ".join([
            str(s.get("name", "")),
            _vec(s.get("mean_rgb")),
            _vec(s.get("p1_rgb")),
            _vec(s.get("p50_rgb")),
            _vec(s.get("p99_rgb")),
            f"{_num(s.get('clip_hi_pct')):.1f}",
            f"{_num(s.get('clip_lo_pct')):.1f}",
            f"{_num(s.get('contrast_std')):.1f}",
            f"{_num(s.get('sat_mean')):.1f}",
            f"{_num(s.get('chroma_mean')):.1f}",
            f"{_num(s.get('wb_rb_ratio'), 1.0):.2f}",
            f"{_num(s.get('exposure_ev_est')):+.2f}",
        ]))
    return "\n".join(lines)


def _build_prompt(stats):
    return (
        "你是一位资深照片调色师。下面是一组同一场景拍摄的照片, 请给出整组统一的"
        "调色方向, 以及每张照片的微调参数。\n\n"
        "数值含义与取值范围 (全部为小幅微调, 不要剧烈改动):\n"
        "- exposure 曝光: EV 补偿, -2.0 ~ 2.0 (正=提亮)\n"
        "- temperature 色温: 开尔文偏移, -2000 ~ 2000 (正=加暖/加黄)\n"
        "- contrast 对比度: -100 ~ 100\n"
        "- highlights 高光: -100 ~ 100 (负=压暗高光)\n"
        "- shadows 阴影: -100 ~ 100 (正=提亮阴影)\n"
        "- saturation 饱和度: -100 ~ 100\n"
        "- vibrance 自然饱和度: -100 ~ 100\n\n"
        "以下客观统计是程序实测得到的真实数值, 请据此判断, 不要凭猜测:\n"
        + _stats_table(stats) + "\n\n"
        "请输出:\n"
        "1. group_style: 一句话概括整组照片的影调与统一调色方向。\n"
        "2. issues: 这一组普遍存在的问题 (曝光/白平衡/高光溢出/死黑等), 中文短句列表。\n"
        "3. suggestions: 整组统一的调色建议, 中文短句列表。\n"
        "4. images: 逐张给出上述 7 个微调数值, 以及一句中文 reason。"
        "name 必须与统计表里的文件名完全一致。"
    )


def _coerce_advice(raw):
    """把模型输出规整成固定结构: 数值转 float, 缺失补 0.0, 字符串转 str。"""
    if not isinstance(raw, dict):
        raise RuntimeError("调色建议不是 JSON 对象")
    out = {
        "group_style": str(raw.get("group_style") or ""),
        "issues": [str(x) for x in (raw.get("issues") or [])],
        "suggestions": [str(x) for x in (raw.get("suggestions") or [])],
        "images": [],
    }
    for item in (raw.get("images") or []):
        if not isinstance(item, dict):
            continue
        row = {"name": str(item.get("name") or "")}
        for key in _NUM_KEYS:
            row[key] = _num(item.get(key))
        row["reason"] = str(item.get("reason") or "")
        out["images"].append(row)
    return out


def _abort(resp):
    """打断阻塞中的流式读取: 只对底层 socket 做 shutdown, 绝不在这里 close。

    踩过的坑: 这里如果调 resp.close(), 读取线程正卡在 BufferedReader.readline 里、
    持有缓冲区锁, close() 会一直等这把锁 —— 于是"取消"自己反被卡住, 要等对端把连接
    关掉、读取线程释放锁才返回 (Windows 上实测能把取消拖 30 秒)。shutdown(SHUT_RDWR)
    不碰缓冲区锁, 立刻返回, 同时让对端 TCP 收到断开而停止生成 token。读取线程是
    daemon, 连接一断它就自己结束了, 不用在这里收。
    """
    try:
        sock = resp.fp.raw._sock
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:  # noqa: BLE001 - 拿不到底层 socket 就当作尽力而为, 调用方会照常抛"已取消"
        pass


def _tail(parts, limit=160):
    """取已累积文本的最后 limit 个字符, 把回车/换行压成空格供 GUI 单行显示。"""
    text = "".join(parts)
    return text[-limit:].replace("\r", " ").replace("\n", " ")


def _chat_stream(payload, model, timeout, on_progress, cancel):
    """流式 POST /api/chat (payload 里 stream 必须为 True), 返回累积的 message.content。

    逐行读 NDJSON: 每行解析出 message.content 片段累积起来, 流结束后由调用方
    把拼接结果当完整 JSON 解析 (与非流式的 message.content 完全一致)。
    on_progress 收到 {"chars": 已累积字符数, "tail": 末尾约 160 字符(换行压成空格)}:
    首次立即回调, 之后约每 _PROGRESS_INTERVAL_S 秒回调一次, 避免刷屏。
    cancel (threading.Event) 置位立即中止并抛 RuntimeError("已取消"): 给了 cancel 时,
    阻塞读取会放进独立线程, 调用线程每 0.05s 醒一次查 cancel, 同时一个看守线程强制
    关闭响应 —— 这样模型慢/卡住时点取消能当场生效, 而不是等下一行数据到来。
    结构化输出被老版本拒绝 (HTTP 400) 时退回 format="json" 重试一次 (同样流式)。
    """
    try:
        req = urllib.request.Request(
            base_url() + "/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        # timeout 是单次读取超时, 不是总时长 —— 流式下每个 read 都会重新计时
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # 老版本 Ollama 不认 JSON Schema, 退回 format="json" 重试 (retry 仍是流式)
        if exc.code == 400 and isinstance(payload.get("format"), dict):
            retry = dict(payload)
            retry["format"] = "json"
            return _chat_stream(retry, model, timeout, on_progress, cancel)
        detail = _read_error_body(exc)
        if exc.code == 404:
            raise RuntimeError(
                f"Ollama 找不到模型 {model}, 请先运行: ollama pull {model}"
            ) from exc
        raise RuntimeError(f"Ollama 请求失败 (HTTP {exc.code}): {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            "无法连接本地 Ollama (127.0.0.1:11434), 请先启动 ollama serve"
        ) from exc

    # finished: 读取结束(正常/异常/取消)时置位; box 把结果或异常回传给调用线程
    finished = threading.Event()
    box = {}

    def _read():
        """在独立线程里跑阻塞的 NDJSON 读取, 结果/异常放进 box。

        为什么读取不放在调用线程: Windows 上对端/本端 shutdown() 唤不醒另一个
        线程里阻塞的 recv (实测 close/shutdown/CancelSynchronousIo 都不行),
        所以调用线程绝不能直接趴在网络上 —— 否则点了取消要等下一行数据到来才醒。
        读取线程是 daemon, 取消后即使还卡在 recv 上, 也不会阻止进程退出。
        """
        parts = []
        chars = 0
        seen_content = False
        last_emit = 0.0     # 0.0 表示尚未回调过 -> 首个片段立即回调
        try:
            # with 保证异常/取消时关闭连接 (终止在途读取)
            with resp:
                # 打开后立刻查一次, 已置位的 cancel 不读任何数据就中止
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("已取消")
                for raw in resp:
                    # 逐行查 cancel (双保险): 看守线程负责打断阻塞, 这里负责及时收手
                    if cancel is not None and cancel.is_set():
                        raise RuntimeError("已取消")
                    line = (raw.decode("utf-8", "replace").strip()
                            if isinstance(raw, bytes) else str(raw).strip())
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(obj, dict):
                        continue
                    err = obj.get("error")
                    if err:
                        raise RuntimeError(f"Ollama 错误: {err}")
                    msg = obj.get("message")
                    added = False
                    if isinstance(msg, dict) and "content" in msg:
                        seen_content = True
                        piece = msg.get("content")
                        if piece:
                            text = str(piece)
                            parts.append(text)
                            chars += len(text)
                            added = True
                    # 只在"有新内容"时回调, 保证 chars 单调递增; 首次立即回调
                    if callable(on_progress) and (added or (last_emit == 0.0 and seen_content)):
                        now = time.monotonic()
                        if last_emit == 0.0 or (now - last_emit) >= _PROGRESS_INTERVAL_S:
                            on_progress({"chars": chars, "tail": _tail(parts)})
                            last_emit = now
                    if obj.get("done"):
                        break
            if not seen_content:
                raise RuntimeError("Ollama 返回里缺少 message.content")
            box["content"] = "".join(parts)
        except RuntimeError as exc:
            # 主动抛出的真实错误 (Ollama 错误 / 已取消 / 缺 content) 原样回传
            box["error"] = exc
        except Exception as exc:  # noqa: BLE001 - 被强制打断的读取归一为中文 RuntimeError
            if cancel is not None and cancel.is_set():
                box["error"] = RuntimeError("已取消")
            else:
                box["error"] = RuntimeError(f"读取 Ollama 流式响应失败: {exc}")
        finally:
            # 无论如何都置位, 看守线程据此退出, 不泄漏
            finished.set()

    if cancel is None:
        # 没有取消需求: 不起多余线程, 直接把读取跑在当前线程 (行为与旧版一致)
        _read()
    else:
        # 有取消需求: 看守线程 + 读取线程
        def _watch():
            """看守线程: 等 finished; 期间若 cancel 置位就强制关闭响应并退出。

            强关是为了让服务端尽快察觉断开、停止继续生成 token (读线程能否被
            唤醒取决于平台, 调用线程另有轮询兜底)。
            """
            while not finished.wait(0.05):
                if cancel.is_set():
                    _abort(resp)
                    return

        threading.Thread(target=_read, name="ollama-read", daemon=True).start()
        threading.Thread(target=_watch, name="ollama-cancel",
                         daemon=True).start()
        # 调用线程不直接阻塞在网络上: 每 0.05s 醒一次, cancel 置位当场抛错,
        # 这样"点取消"能立即生效, 而不是等模型吐出下一行才反应。
        while not finished.wait(0.05):
            if cancel.is_set():
                _abort(resp)
                raise RuntimeError("已取消")

    if "error" in box:
        raise box["error"]
    return box["content"]


def analyze_images(images, stats, model=None, timeout=None,
                   on_progress=None, cancel=None):
    """把 JPEG 字节 + 客观统计发给本地视觉模型, 返回校验后的调色建议 dict。

    model / timeout 为空时分别用配置里的默认模型 (default_model()) 与超时
    (request_timeout())。请求走流式 (stream=True): on_progress 回调实时进度
    ({"chars": 已累积字符数, "tail": 末尾约 160 字符, 换行已压成空格}),
    cancel (threading.Event) 置位即可中止 (抛 RuntimeError("已取消"));
    即使模型卡住不吐字, 调用线程也会在 ~0.05s 内立刻抛错 (看守线程同时断开连接)。
    """
    model = model or default_model()
    timeout = timeout or request_timeout()
    if not images:
        raise RuntimeError("没有可分析的图片")

    # 裸 base64, 不带 data:image/... 前缀 (Ollama /api/chat 的约定)
    b64 = [base64.b64encode(img).decode("ascii") for img in images]
    message = {"role": "user", "content": _build_prompt(stats), "images": b64}
    payload = {
        "model": model,
        "messages": [message],
        "stream": True,
        "format": ADVICE_SCHEMA,
        "options": {"temperature": 0},
    }

    content = _chat_stream(payload, model, timeout, on_progress, cancel)
    try:
        raw = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Ollama 返回内容不是合法 JSON: {exc}\n原始内容: {content[:500]}"
        ) from exc
    return _coerce_advice(raw)
