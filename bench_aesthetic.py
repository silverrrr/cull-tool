"""实测美感模型的实际开销 —— 在**你要跑的那台机器**上跑这个, 别照搬别人的数字。

测每个模型: 单张耗时 (可选多种 onnxruntime 线程数)、进程内存增量、以及分数分布。
模型文件不在本地时会顺手下载 (aes_v25 约 1.7GB)。

用法 (在项目目录下):
    venv\\Scripts\\python.exe bench_aesthetic.py "E:\\照片\\某批" --models nima aes_v25
    venv\\Scripts\\python.exe bench_aesthetic.py "E:\\照片\\某批" --models aes_v25 --threads 4
    venv\\Scripts\\python.exe bench_aesthetic.py --models aes_v25 --onnx 某个.onnx   # 跳过下载

每台机器跑一遍, 就知道该选哪个模型、并行进程数该设多少。
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                             # noqa: E402
import cv2                                                     # noqa: E402
import cull                                                    # noqa: E402

PHOTO_PATTERNS = ("*.ARW", "*.arw", "*.CR2", "*.cr2", "*.NEF", "*.nef",
                  "*.RAF", "*.raf", "*.JPG", "*.jpg")


def rss_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / 1e6
    except Exception:                                          # noqa: BLE001
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:                                          # noqa: BLE001
        return 0.0


def find_photos(folder, n):
    out = []
    for pat in PHOTO_PATTERNS:
        out += glob.glob(os.path.join(folder, pat))
    out = sorted(out)
    if not out:
        sys.exit(f"{folder} 里没找到照片")
    if len(out) > n:
        idx = np.linspace(0, len(out) - 1, n).astype(int)
        out = [out[i] for i in idx]
    return out


def load_imgs(paths):
    imgs, t = [], time.time()
    for p in paths:
        bgr, _ = cull.decode(p, False)
        imgs.append(bgr)
    print(f"解码 {len(imgs)} 张用了 {time.time() - t:.1f}s "
          f"({(time.time() - t) / len(imgs) * 1000:.0f} ms/张)")
    return imgs


def bench(model_key, imgs, threads, onnx_path=None):
    import onnxruntime as ort
    path = onnx_path or cull.ensure_asset(model_key)
    print(f"模型文件: {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads or 0
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    t = time.time()
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    load = time.time() - t
    rss0 = rss_mb()
    cull.aesthetic_score(imgs[0], sess, model_key)              # 预热
    t = time.time()
    scores = [cull.aesthetic_score(x, sess, model_key) for x in imgs]
    dt = (time.time() - t) / len(imgs)
    rss = rss_mb()
    s = np.array(scores)
    print(f"  threads={threads or 'auto'}  加载 {load:.1f}s  {dt * 1000:.0f} ms/张  "
          f"内存 +{rss - rss0:.0f} MB  分数 {s.min():.2f}~{s.max():.2f} (mean {s.mean():.2f})")
    return dt, rss - rss0


def main():
    ap = argparse.ArgumentParser(description="实测美感模型的耗时与内存")
    ap.add_argument("folder", nargs="?", default=None, help="照片目录")
    ap.add_argument("--models", nargs="+", default=["nima", "aes_v25"],
                    choices=list(cull.AESTHETIC_MODELS))
    ap.add_argument("--n", type=int, default=20, help="抽样张数 (默认 20)")
    ap.add_argument("--threads", type=int, nargs="+", default=[4],
                    help="onnxruntime 线程数, 可给多个 (默认 4)")
    ap.add_argument("--onnx", default=None, help="直接指定模型文件, 跳过下载")
    args = ap.parse_args()

    folder = args.folder or os.getcwd()
    imgs = load_imgs(find_photos(folder, args.n))

    import onnxruntime as ort                                   # noqa: F401
    print(f"onnxruntime {ort.__version__}   机器: "
          f"{os.cpu_count()} 逻辑核")

    for key in args.models:
        if args.onnx and len(args.models) == 1:
            bench(key, imgs, args.threads[0], args.onnx)
            continue
        for th in args.threads:
            try:
                bench(key, imgs, th)
            except Exception as exc:                           # noqa: BLE001
                print(f"  [!] {key} 跑不起来: {exc}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
