#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选图工具 · 桌面客户端

单个 WebView2 窗口 (pywebview + pythonnet): 选文件夹 -> 分析 -> 页面里看结果、
筛选、勾选 -> 直接把勾选的文件搬到目标文件夹。

**默认不写任何文件** —— 报告只在窗口里显示。想存 CSV / 清单得点「导出」。

用法:
    选图工具.exe                    # 打开界面
    选图工具.exe "D:\\照片\\2026-01"  # 带上文件夹直接启动
"""

import base64
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if not getattr(sys, "frozen", False):
    sys.path.insert(0, HERE)

import webview

import cull  # 本地模块, 打包时会被一起收进去

APP_TITLE = "选图工具"
UI_HTML = os.path.join(HERE, "ui.html")
BATCH = 60                # 每次从 Python 拿多少张卡片 (缩略图 base64, 太多会卡)
BIG_SIDE = 1600           # 点开看的大图最长边
BIG_CACHE_MAX = 64        # 大图内存 LRU 上限, 约 22MB


# ==========================================================================
# 界面可以调的接口 (window.pywebview.api.xxx)
# ==========================================================================
class _Cancelled(Exception):
    """用户点了取消。不是错误, 走专门的处理分支。"""


class Api:
    """给页面调用的接口。所有耗时操作都在后台线程, 不卡界面。"""

    def __init__(self):
        self.state = {}                 # folder -> rows
        self.queue_dirs = []            # 待分析
        self.done_dirs = []             # 已分析
        self.cur = None                 # 当前在看的文件夹
        self.q = queue.Queue()          # 给界面轮的进度消息
        self.big = OrderedDict()        # folder/name -> 大图 base64
        self._big_lock = threading.Lock()
        self._jobs = ThreadPoolExecutor(max_workers=4)   # 大图解码
        self._model = None
        self._run_id = 0                # 每跑一轮 +1, 用来作废旧任务的消息
        self._cancel = threading.Event()

    # ---- 文件夹选择 ----
    def pick_folders(self):
        r = webview.windows[0].create_file_dialog(webview.FileDialog.FOLDER, allow_multiple=True)
        return [str(x) for x in r] if r else []

    def pick_dest(self):
        r = webview.windows[0].create_file_dialog(webview.FileDialog.FOLDER)
        return str(r[0]) if r else ""

    def add_folders(self, dirs):
        for d in dirs or []:
            if d not in self.queue_dirs and d not in self.done_dirs and d not in self.state:
                self.queue_dirs.append(d)
        return self.get_queue()

    def get_queue(self):
        """页面启动时拉一次, 这样命令行带进来的文件夹也能显示出来。"""
        return {"queue": list(self.queue_dirs), "done": list(self.done_dirs)}

    def remove_folder(self, path):
        """界面上的 × —— 必须同时改 Python 这边, 否则删不掉又加不回来。"""
        if path in self.queue_dirs:
            self.queue_dirs.remove(path)
        if path in self.done_dirs:
            self.done_dirs.remove(path)
        self.state.pop(path, None)
        if self.cur == path:
            self.cur = None
        with self._big_lock:                       # 这个文件夹的大图缓存也一起扔
            for k in [k for k in self.big if k.startswith(path + "|")]:
                del self.big[k]
        return self.get_queue()

    # ---- 分析 ----
    def analyze(self, full=False, jobs=4):
        """起后台线程, 立即返回。界面靠 poll() 拿进度。"""
        if not self.queue_dirs:
            return {"ok": False, "error": "先选一个照片文件夹"}
        self._cancel.clear()
        self._run_id += 1
        threading.Thread(target=self._run_queue,
                         args=(list(self.queue_dirs), full, jobs, self._run_id),
                         daemon=True).start()
        return {"ok": True, "n": len(self.queue_dirs)}

    def cancel(self):
        """中止这一轮: 已经算出来的结果全丢, 但文件夹留在队列里, 可以直接重跑。"""
        # 只置标志, 不动 _run_id —— 递增会让后台线程以为任务过期直接走人,
        # 那样它就不会发 idle, 界面会一直等在"分析中"。
        # queue_dirs 一个都不动, 由 _run_queue 退出时把没跑完的放回去。
        self._cancel.set()
        self.state.clear()
        self.done_dirs.clear()
        self.cur = None
        with self._big_lock:
            self.big.clear()
        return {"ok": True, "queue": list(self.queue_dirs)}

    def _cancelled(self, run_id):
        return self._cancel.is_set() or self._run_id != run_id

    def _model_path(self):
        # 首次运行要联网下 230KB 模型, 所以等到真要点分析时才加载, 别卡住窗口创建
        if self._model is None:
            self._model = cull.ensure_model(None)
        return self._model

    def _run_queue(self, folders, full, jobs, run_id):
        try:
            for d in folders:
                if self._cancelled(run_id):
                    break
                if d in self.queue_dirs:
                    self.queue_dirs.remove(d)
                self.q.put(("log", f"开始分析: {os.path.basename(d)}"))
                try:
                    rows = self._analyze_one(d, full=full, jobs=jobs, run_id=run_id)
                    if self._cancelled(run_id):
                        break
                    self.state[d] = rows
                    self.done_dirs.append(d)
                    self.cur = d
                    self.q.put(("done", d))
                except _Cancelled:
                    break
                except RuntimeError as exc:                # 像"没有照片"这种用户能懂的
                    self.q.put(("error", (d, str(exc))))
                except Exception:                          # noqa: BLE001
                    self.q.put(("error", (d, traceback.format_exc())))
            if self._cancel.is_set() and self._run_id == run_id:
                # 没跑完的放回队列, 界面上直接就能再点一次"开始分析"
                rest = [d for d in folders if d not in self.done_dirs]
                for d in rest:
                    if d not in self.queue_dirs:
                        self.queue_dirs.append(d)
                self.q.put(("cancelled", rest))
        finally:
            if self._run_id == run_id:
                self.q.put(("idle", ""))

    def _analyze_one(self, folder, full=False, jobs=4, run_id=0):
        names, _ = cull.list_photos(folder)
        if not names:
            raise RuntimeError("这个文件夹里没有可分析的照片"
                               "(支持 RAW 以及 jpg/jpeg/png/webp/bmp/tif/tiff)")
        n_all = len(names)

        model = self._model_path()
        # 注意: 不建 out_dir, 也不传 preview_dir —— 整个过程一个文件都不写
        cull.worker_init(model, full, None, 0, 1 if jobs > 1 else 0)
        rows = []
        if jobs > 1 and len(names) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=jobs, initializer=cull.worker_init,
                                     initargs=(model, full, None, 0, 1)) as ex:
                futs = {ex.submit(cull.process_one, n, folder): n for n in names}
                done = 0
                for f in as_completed(futs):
                    if self._cancelled(run_id):
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise _Cancelled()
                    done += 1
                    rec = f.result()
                    rows.append(rec)
                    self.q.put(("log", f"[{done}/{n_all}] {rec['file']}  "
                                        f"眼锐度={rec['eye_sharp']:.0f}  脸={rec['faces']}"))
        else:
            for i, n in enumerate(names, 1):
                if self._cancelled(run_id):
                    raise _Cancelled()
                rec = cull.process_one(n, folder)
                rows.append(rec)
                self.q.put(("log", f"[{i}/{n_all}] {rec['file']}  "
                                    f"眼锐度={rec['eye_sharp']:.0f}  脸={rec['faces']}"))

        order = {n: i for i, n in enumerate(names)}       # 完成顺序是乱的, 排回去
        rows.sort(key=lambda r: order.get(r["file"], 0))
        cull.compute_flags(rows)
        return rows

    def poll(self):
        msgs = []
        while True:
            try:
                msgs.append(self.q.get_nowait())
            except queue.Empty:
                break
        return msgs

    # ---- 结果给页面 ----
    def get_batch(self, folder, off=0, limit=BATCH):
        """一次给一批卡片的展示数据 (缩略图已经是 base64 data URI)。"""
        rows = self._rows(folder)
        ordered = sorted(rows, key=lambda r: (r.get("subject") != "face",
                                              -r.get("compare_value", 0)))
        part = ordered[off:off + limit]
        cards = []
        for r in part:
            gs = r.get("group_size", 1)
            cards.append({
                "name": r["file"],
                "thumb": r.get("_thumb") or "",
                "score": (f'眼部锐度 {r["eye_sharp"]}' if r.get("subject") == "face"
                          else f'中央区锐度 {r.get("face_sharp", 0)}'),
                "ratio": (f' · 占组内最佳 {r["ratio_group"]:.0%}'
                          if gs > 1 and r.get("ratio_group") else ""),
                "badge": (f'组{r["group"]} 最清晰' if r.get("best_in_group")
                          else (f'组{r["group"]} 疑似模糊' if r.get("soft")
                                else (f'组{r["group"]}' if gs > 1
                                      else ("无脸" if r.get("subject") == "center" else "")))),
                "meta": " · ".join(x for x in [
                    f'组{r["group"]}({gs}张)' if gs > 1 else "",
                    "中央区(无脸)" if r.get("subject") == "center" else "眼睛对焦",
                    f'脸 {r.get("face_w",0)}px' if r.get("face_w") else "",
                    f'AF {r["af_area"]}' if r.get("af_area") else "",
                    f'ISO {r["iso"]}' if r.get("iso") else "",
                    cull.fmt_exposure(r.get("exposure_time")),
                    f'f/{cull.fmt_rat(r.get("f_number"),1)}' if r.get("f_number") else "",
                    f'{cull.fmt_rat(r.get("focal_length"),0)}mm' if r.get("focal_length") else "",
                ] if x),
                # 筛选用的字段
                "sharp": r.get("compare_value", 0),
                "ratio_v": r.get("ratio_group", 0),
                "best": 1 if r.get("best_in_group") else 0,
                "multi": 1 if gs > 1 else 0,
                "face": 1 if r.get("subject") == "face" else 0,
                "area": str(r.get("af_area", "")),
                "cls": ("best" if r.get("best_in_group")
                        else ("soft" if r.get("soft") else "")),
            })
        areas = sorted({str(r.get("af_area", "")) for r in rows if r.get("af_area")})
        return {
            "cards": cards,
            "total": len(ordered),
            "off": off,
            "more": off + len(part) < len(ordered),
            "folder": folder,
            "basename": os.path.basename(folder),
            "areas": areas,
            "n_face": sum(1 for r in rows if r.get("faces")),
            "n_soft": sum(1 for r in rows if r.get("soft")),
            "n_dup": sum(1 for r in rows if r.get("group_size", 1) > 1),
        }

    def get_big(self, folder, name):
        """点开看大图。解码 ~0.3s, 结果进内存 LRU, 翻回来就是瞬开。"""
        key = folder + "|" + name
        with self._big_lock:
            if key in self.big:
                self.big.move_to_end(key)
                return {"img": self.big[key], "name": name, "cached": True}
        try:
            raw = self._jobs.submit(cull.make_big_jpeg,
                                    os.path.join(folder, name), BIG_SIDE).result()
        except Exception as exc:                              # noqa: BLE001
            return {"error": f"打不开这张: {exc}"}
        if not raw:
            return {"error": "这张解码失败了"}
        data = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        with self._big_lock:
            self.big[key] = data
            while len(self.big) > BIG_CACHE_MAX:
                self.big.popitem(last=False)
        return {"img": data, "name": name, "cached": False}

    def show_in_explorer(self, folder, name):
        """在资源管理器里选中这个原文件。"""
        try:
            subprocess.Popen(["explorer", "/select,", os.path.join(folder, name)])
            return {"ok": True}
        except Exception as exc:                              # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # ---- 移动 ----
    def move(self, folder, names, dest, copy=False, with_jpg=False, preview_dir=""):
        if not dest or not os.path.isdir(dest):
            return {"ok": False, "error": "目标文件夹不存在"}
        if not names:
            return {"ok": False, "error": "还没有勾选照片"}
        todo, skipped = cull.plan_move(folder, names, dest, with_jpg)
        mb = round(sum(os.path.getsize(s) for s, _ in todo if os.path.isfile(s)) / 1e6, 1)
        # 预览必须在搬文件**之前**做: 勾了"移动"的话, 搬完源文件就没了。
        pv = None
        if preview_dir:
            pv = self._write_previews(folder, names, preview_dir)
        moved, failed = cull.do_move(folder, names, dest, copy=copy,
                                     with_jpg=with_jpg, dry_run=False, out_dir=None)
        msg = (f'已{"复制" if copy else "移动"} {moved} 个文件'
               + (f", 跳过 {len(skipped)} 个" if skipped else "")
               + (f", 失败 {failed} 个" if failed else ""))
        if pv and pv["n"]:
            msg += f"；预览图 {pv['n']} 张 -> {os.path.basename(preview_dir)}"
        return {"ok": failed == 0, "moved": moved, "failed": failed,
                "skipped": len(skipped), "mb": mb,
                "copied": bool(copy), "preview": pv, "msg": msg}

    def _write_previews(self, folder, names, preview_dir, side=1600):
        """给挑中的每张照片写一张预览 jpg, 集中放到 preview_dir。"""
        os.makedirs(preview_dir, exist_ok=True)
        n = fail = 0
        for i, name in enumerate(names, 1):
            src = os.path.join(folder, name)
            if not os.path.isfile(src):
                fail += 1
                continue
            dst = os.path.join(preview_dir, cull.preview_name(name))
            try:
                cull.make_preview_jpeg(src, dst, side=side)
                n += 1
            except Exception:                                 # noqa: BLE001
                fail += 1
            self.q.put(("log", f"生成预览 {i}/{len(names)}  {os.path.basename(dst)}"))
        return {"n": n, "failed": fail, "dir": preview_dir}

    # ---- 导出 (默认不写, 只有点了才写) ----
    def export_csv(self, folder):
        rows = self._rows(folder)
        if not rows:
            return {"ok": False, "error": "这个文件夹还没分析"}
        r = webview.windows[0].create_file_dialog(
            webview.FileDialog.SAVE, save_filename="选图报告.csv",
            file_types=("CSV (*.csv)",))
        if not r:
            return {"ok": False, "cancelled": True}
        path = str(r[0])
        if not path.lower().endswith(".csv"):
            path += ".csv"
        cull.write_csv(rows, path)
        return {"ok": True, "path": path}

    def export_list(self, folder, names):
        """把当前勾选的文件名写成一个文本清单。"""
        if not names:
            return {"ok": False, "error": "还没有勾选照片"}
        r = webview.windows[0].create_file_dialog(
            webview.FileDialog.SAVE, save_filename="选图清单.txt",
            file_types=("文本 (*.txt)",))
        if not r:
            return {"ok": False, "cancelled": True}
        path = str(r[0])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        return {"ok": True, "path": path, "n": len(names)}

    def _rows(self, folder):
        rows = self.state.get(folder)
        if not rows:
            raise RuntimeError("这个文件夹还没分析, 或者已经被移走了")
        return rows


# ==========================================================================
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                        # noqa: BLE001
        pass

    # pywebview 在 import 时就选渲染后端, 没装 WebView2 会静默退回已废弃的 IE 内核
    import webview.platforms.winforms as _winforms
    if getattr(_winforms, "renderer", "") != "edgechromium":
        import tkinter.messagebox as mb
        mb.showerror(APP_TITLE,
                     "这台电脑没有 Microsoft Edge WebView2 运行时, 界面跑不起来。\n\n"
                     "请到微软官网下载安装 (Win11 一般自带):\n"
                     "https://developer.microsoft.com/microsoft-edge/webview2/")
        return 1

    if not os.path.isfile(UI_HTML):
        raise SystemExit(f"找不到界面文件: {UI_HTML}")

    api = Api()
    webview.create_window(APP_TITLE, html=read_ui(), js_api=api,
                          width=1280, height=860, min_size=(900, 620),
                          background_color="#18181b")
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        api.add_folders([sys.argv[1]])
    webview.start(debug=False, private_mode=False)
    return 0


def read_ui():
    with open(UI_HTML, "r", encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    # Windows 上多进程用 spawn 启动子进程; 打包成 exe 后必须先调这个,
    # 否则子进程会重新执行整个模块并无限递归
    multiprocessing.freeze_support()
    sys.exit(main())
