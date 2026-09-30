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
import ollama_advise  # 本地模块: 调本机 Ollama 出调色建议 (可能连不上, 要能容错)

APP_TITLE = "选图工具"
UI_HTML = os.path.join(HERE, "ui.html")
BATCH = 60                # 每次从 Python 拿多少张卡片 (缩略图 base64, 太多会卡)
BIG_SIDE = 1600           # 点开看的大图最长边
BIG_CACHE_MAX = 64        # 大图内存 LRU 上限, 约 22MB


def _even_indices(n, k):
    """从 0..n-1 里均匀挑 k 个下标, 保留顺序、含首尾。"""
    if k >= n:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


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
        self.advice = {}                # folder -> 调色建议 payload
        self._advice_id = 0             # 每问一次 +1, 用来作废过期的建议任务
        self._advice_cancel = threading.Event()  # 调色建议的取消标志 (独立于 _cancel / _pull_cancel)
        self.advice_model = ollama_advise.default_model()  # 当前选用的调色建议模型 (跟随本地配置)
        self._pull_cancel = threading.Event()             # 拉模型时的取消标志 (独立于 _cancel)
        self._pulling = False                             # 是否正在后台拉取模型
        self._cancel = threading.Event()
        self._pool_lock = threading.Lock()
        self._pool = None               # 当前这轮的进程池, 取消时要 terminate 掉
        self._last_full = False         # 上一轮是不是全分辨率, 换主体时要用

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
    def analyze(self, full=False, jobs=4, force=False):
        """起后台线程, 立即返回。界面靠 poll() 拿进度。

        force=True 时忽略结果缓存, 强制重新解码这一轮。

        要分析的文件夹 = 队列里的 + **已经分析过的**。第一遍跑完文件夹会从
        queue_dirs 挪进 done_dirs, 而界面上「开始分析」对已完成的文件夹是
        允许再点的 (updateRun 只排除 state==='run')。所以这里必须把 done_dirs
        一起算进来, 否则第二次点会返回"先选一个照片文件夹", 后台根本没起,
        界面就永远停在"正在分析…"且取消按钮点不动。
        """
        targets = [d for d in self.queue_dirs if d not in self.done_dirs]
        targets += [d for d in self.done_dirs if d not in self.queue_dirs]
        if not targets:
            return {"ok": False, "error": "先选一个照片文件夹"}
        self._cancel.clear()
        self._run_id += 1
        threading.Thread(target=self._run_queue,
                         args=(list(targets), full, jobs, self._run_id, bool(force)),
                         daemon=True).start()
        return {"ok": True, "n": len(targets)}

    def cancel(self):
        """中止这一轮: 已经算出来的结果全丢, 但文件夹留在队列里, 可以直接重跑。"""
        # 只置标志, 不动 _run_id —— 递增会让后台线程以为任务过期直接走人,
        # 那样它就不会发 idle, 界面会一直等在"分析中"。
        # queue_dirs 一个都不动, 由 _run_queue 退出时把没跑完的放回去。
        self._cancel.set()
        self._advice_id += 1            # 作废在跑的调色建议任务 (和 _run_id 分开)
        self._kill_pool()
        self.state.clear()
        self.done_dirs.clear()
        self.cur = None
        with self._big_lock:
            self.big.clear()
        return {"ok": True, "queue": list(self.queue_dirs)}

    def _kill_pool(self):
        """terminate 掉正在跑的 worker 进程。

        只置 _cancel 标志是不够的: 已经在解码的那几张不会因为标志停下来,
        worker 会一直把整批照片跑完, 几个进程各占几百 MB 内存继续读盘。
        terminate 是当场掐断, 只丢内存里的中间结果, 不动源文件。
        """
        with self._pool_lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.stop()

    def _cancelled(self, run_id):
        return self._cancel.is_set() or self._run_id != run_id

    def _model_path(self):
        # 首次运行要联网下 230KB 模型, 所以等到真要点分析时才加载, 别卡住窗口创建
        if self._model is None:
            self._model = cull.ensure_model(None)
        return self._model

    def _run_queue(self, folders, full, jobs, run_id, force=False):
        # 明确记下"这轮真的处理完了哪些", 不能靠反推:
        # 循环开头就把 d 从 queue_dirs 里摘掉了, 所以"不在 queue_dirs"并不等于
        # "已完成" —— 被取消的那个同样不在队列里, 会被误判成已完成而丢掉。
        handled = set()
        try:
            for d in folders:
                if self._cancelled(run_id):
                    break
                if d in self.queue_dirs:
                    self.queue_dirs.remove(d)
                # 重新分析一个已完成的文件夹时, 它本来就在 done_dirs 里,
                # 再 append 一次就会重复, 于是第二遍 targets 里出现同一个路径两遍。
                if d in self.done_dirs:
                    self.done_dirs.remove(d)
                self.q.put(("log", f"开始分析: {os.path.basename(d)}"))
                try:
                    rows = self._analyze_one(d, full=full, jobs=jobs, run_id=run_id,
                                             force=force)
                    if self._cancelled(run_id):
                        break
                    self.state[d] = rows
                    self.done_dirs.append(d)
                    self.cur = d
                    handled.add(d)
                    self.q.put(("done", d))
                except _Cancelled:
                    break
                except RuntimeError as exc:                # 像"没有照片"这种用户能懂的
                    handled.add(d)
                    self.q.put(("error", (d, str(exc))))
                except Exception:                          # noqa: BLE001
                    handled.add(d)
                    self.q.put(("error", (d, traceback.format_exc())))

            if self._cancel.is_set() and self._run_id == run_id:
                # 没跑完的放回队列, 界面上直接就能再点一次"开始分析"
                rest = [d for d in folders if d not in handled]
                for d in rest:
                    if d not in self.queue_dirs:
                        self.queue_dirs.append(d)
                self.q.put(("cancelled", rest))
        finally:
            if self._run_id == run_id:
                self.q.put(("idle", ""))

    def _analyze_one(self, folder, full=False, jobs=4, run_id=0, force=False):
        names, _ = cull.list_photos(folder)
        if not names:
            raise RuntimeError("这个文件夹里没有可分析的照片"
                               "(支持 RAW 以及 jpg/jpeg/png/webp/bmp/tif/tiff)")
        n_all = len(names)

        # 有缓存就直接用, 不逐张解码。校验过期的活儿在 read_cache 里。
        # 注意要在 _model_path() 之前: 命中缓存时根本不需要人脸检测模型,
        # 第一次跑也就不用为了看一眼结果去联网下那 230KB。
        if not force:
            hit = cull.read_cache(folder, full, names=names)
            if hit is not None:
                self.q.put(("log", f"读取缓存 ({len(hit)} 张, 跳过解码): "
                                    f"{os.path.basename(folder)}"))
                return hit

        model = self._model_path()
        # 注意: 不建 out_dir, 也不传 preview_dir —— 除结果缓存外一个文件都不写
        self._last_full = bool(full)
        # worker 是 spawn 出来的, 读不到本进程对 _CFG 的改动, 美感模型必须显式传
        aes_model = cull.aesthetic_model_name()
        cull.worker_init(model, full, None, 0, 1 if jobs > 1 else 0, aes_model)
        rows = []
        if jobs > 1 and len(names) > 1:
            from concurrent.futures import as_completed
            ex = cull.HardStopPool(max_workers=jobs, initializer=cull.worker_init,
                                   initargs=(model, full, None, 0, 1, aes_model))
            with self._pool_lock:
                self._pool = ex
            try:
                futs = {ex.submit(cull.process_one, n, folder): n for n in names}
                done = 0
                for f in as_completed(futs):
                    # 取消可能只置了标志还没来得及 terminate (或者 terminate 的
                    # 结果还没传回来), 所以这里也要看一眼, 否则会把整批解码完
                    # 才返回 —— 那正是最初"取消后还在读照片"的原因。
                    if self._cancelled(run_id):
                        break
                    rec = f.result()
                    done += 1
                    rows.append(rec)
                    self.q.put(("log", f"[{done}/{n_all}] {rec['file']}  "
                                        f"眼锐度={rec['eye_sharp']:.0f}  脸={rec['faces']}"))
            except BaseException:
                # worker 被 terminate 之后, 池子会变成 broken, 在途的 future
                # 抛的是 BrokenProcessPool 而不是 CancelledError —— 所以这里
                # 统一按"是不是用户取消"来判断该报什么, 免得弹一屏 traceback。
                ex.stop()
                if self._cancelled(run_id):
                    raise _Cancelled() from None
                raise
            else:
                # 正常跑完, 或者上面 break 出来的: 都要把 worker 收干净,
                # 否则已 terminate 的进程会一直挂在进程表里占内存。
                ex.stop()
            finally:
                with self._pool_lock:
                    if self._pool is ex:
                        self._pool = None
            if self._cancelled(run_id):
                raise _Cancelled()
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
        # 存缓存给下次用。写失败顶多少一次秒开, 不该让这轮结果出不来,
        # 所以 write_cache 内部已经吞掉异常, 这里不必再包一层。
        cull.write_cache(folder, rows, full)
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
    def _order_rows(self, rows, sort):
        """三种排序。group/name 两种会让同组的照片**连续**排在一起。

        "sharp"  : 旧行为 —— 按锐度降序, 不分节 (默认, 保持向后兼容)
        "group"  : 多张组集中置顶, 组内按锐度降序 (组内最佳排第一)
        "name"   : 按文件名走, 但同组的聚成一块 (组在它最早那张的位置上)
        """
        if sort == "group":
            byg = {}
            for r in rows:
                if r.get("group_size", 1) > 1:
                    byg.setdefault(r["group"], []).append(r)
            units = []
            for gid, g in byg.items():
                g.sort(key=lambda r: -r.get("compare_value", 0))
                units.append((max(r.get("compare_value", 0) for r in g), gid, g))
            units.sort(key=lambda t: (-t[0], t[1]))          # 强的组排前面
            out = []
            for _, _, g in units:
                out.extend(g)
            out.extend(sorted((r for r in rows if r.get("group_size", 1) <= 1),
                              key=lambda r: -r.get("compare_value", 0)))
            return out

        if sort == "name":
            units = {}
            for r in rows:
                # 多张组当一个整体, 单张各自成一个单位
                key = ("g", r["group"]) if r.get("group_size", 1) > 1 else ("f", r["file"])
                units.setdefault(key, []).append(r)
            out = []
            for _, g in sorted(units.items(),
                               key=lambda it: min(x["file"] for x in it[1]).lower()):
                g.sort(key=lambda r: -r.get("compare_value", 0))
                out.extend(g)
            return out

        if sort == "value":
            return sorted(rows, key=lambda r: (r.get("edit_value") is None,
                                               -(r.get("edit_value") or 0)))

        return sorted(rows, key=lambda r: (r.get("subject") != "face",
                                           -r.get("compare_value", 0)))

    def get_batch(self, folder, off=0, limit=BATCH, sort="sharp"):
        """一次给一批卡片的展示数据 (缩略图已经是 base64 data URI)。"""
        rows = self._rows(folder)
        ordered = self._order_rows(rows, sort)
        part = ordered[off:off + limit]
        # 分节模式下**不能把一组切成两页** —— 否则组标题会在两页各出现一次,
        # 而且同组的照片会隔着屏幕。碰到切中的就把这一组剩下的全带上。
        if sort in ("group", "name") and part:
            gs = part[-1].get("group_size", 1)
            if gs > 1:
                gid = part[-1]["group"]
                end = off + len(part)
                while end < len(ordered) and ordered[end]["group"] == gid:
                    end += 1
                part = ordered[off:end]
        cards = []
        for r in part:
            gs = r.get("group_size", 1)
            cards.append({
                "name": r["file"],
                "value": r.get("edit_value"),
                "parts": r.get("score_parts") or {},
                "yaw": r.get("yaw"),
                "ear": r.get("ear"),
                "mos_tech": r.get("mos_tech"),
                "aes": r.get("aes"),
                "thumb": r.get("_thumb") or "",
                "group": r.get("group", 0),
                "group_size": gs,
                "score": (f'眼部锐度 {r["eye_sharp"]}' if r.get("subject") == "face"
                          else f'中央区锐度 {r.get("face_sharp", 0)}'),
                "ratio": (f' · 占组内最佳 {r["ratio_group"]:.0%}'
                          if gs > 1 and r.get("ratio_group") else ""),
                "badge": (f'组{r["group"]} 最清晰' if r.get("best_in_group")
                          else (f'组{r["group"]} 疑似模糊' if r.get("soft")
                                else (f'组{r["group"]}' if gs > 1
                                      else ("无脸" if r.get("subject") == "center" else "")))),
                # 主体可能不是同一个人 (多脸合影里"最清楚那张"会在人之间跳),
                # 这种组的组内锐度不可比, 标出来提醒
                "warn": "⚠" if r.get("subject_uncertain") else "",
                "uncertain": 1 if r.get("subject_uncertain") else 0,
                # 检出多张脸时, 卡片上给个"换主体"的下拉
                "n_faces": r.get("faces", 0),
                "subject_idx": r.get("subject_idx", -1),
                "face_metrics": (r.get("_face_metrics") or [])[:12],
                "meta": " · ".join(x for x in [
                    (f'修图价值 {r["edit_value"]:.0f}' if r.get("edit_value") is not None else ""),
                    f'组{r["group"]}({gs}张)' if gs > 1 else "",
                    "中央区(无脸)" if r.get("subject") == "center" else "眼睛对焦",
                    (f'主体脸 {r["face_w"]}px' if (r.get("faces") or 0) > 1
                     else f'脸 {r.get("face_w",0)}px') if r.get("face_w") else "",
                    f'共{r["faces"]}张脸' if (r.get("faces") or 0) > 1 else "",
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
                "multi_face": 1 if (r.get("faces") or 0) > 1 else 0,
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

    def get_face_choices(self, folder, name):
        """列出这张照片检出的所有脸, 供界面下拉让你选主体。

        只读缓存里的 _face_metrics, 不重新解码 —— 列表能秒出,
        真的换主体时 (set_subject) 才重解码那一张。
        """
        rows = self._rows(folder)
        row = next((r for r in rows if r["file"] == name), None)
        if row is None:
            return {"faces": []}
        metrics = row.get("_face_metrics") or []
        cur = row.get("subject_idx", 0)
        out = []
        for m in metrics:
            out.append({
                "idx": m.get("idx"),
                "label": (f"#{m.get('idx', 0) + 1}  眼锐度 {m.get('eye', 0):.0f}"
                          f"  脸宽 {m.get('w', 0)}px"
                          f"  位置 ({m.get('cx', 0):.2f},{m.get('cy', 0):.2f})"),
                "eye": m.get("eye", 0),
                "sel": 1 if m.get("idx") == cur else 0,
            })
        return {"faces": out, "current": cur, "n_faces": row.get("faces", 0)}

    def set_subject(self, folder, name, idx):
        """手动把某张照片的主体换成第 idx 张脸, 并重算分组。

        会按需重新解码这一张 (~0.3s) —— sig / 脸部锐度都依赖原图像素。
        换完立刻写回缓存, 所以下次读缓存就是新主体。
        """
        rows = self._rows(folder)
        row = next((r for r in rows if r["file"] == name), None)
        if row is None:
            return {"ok": False, "error": "这张照片不在当前结果里"}
        try:
            cull.worker_init(self._model_path(), self._last_full, None, 0, 0,
                             cull.aesthetic_model_name())
            cull.reselect_subject(row, name, folder, int(idx),
                                  full=self._last_full, af_point=None)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:                              # noqa: BLE001
            return {"ok": False, "error": f"换主体失败: {exc}"}
        # 分组依赖 sig, 主体变了必须整体重算
        cull.compute_flags(rows)
        try:
            cull.write_cache(folder, rows, self._last_full)
        except Exception:                                   # noqa: BLE001
            pass
        return {"ok": True, "name": name, "subject_idx": int(idx),
                "eye_sharp": row.get("eye_sharp", 0),
                "group": row.get("group", 0),
                "n_soft": sum(1 for r in rows if r.get("soft"))}

    def show_in_explorer(self, folder, name):
        """在资源管理器里选中这个原文件。"""
        try:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                return {"ok": False, "error": "文件不在了(可能已经移走)"}
            # explorer /select, 的逗号是它自己的语法; 路径带引号 explorer 才认
            subprocess.Popen(["explorer", "/select,", os.path.abspath(path)])
            return {"ok": True}
        except Exception as exc:                              # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def open_original(self, folder, name):
        """用系统默认程序打开原图 (双击文件的效果)。

        用 os.startfile 而不是 subprocess, 它才是 Windows 官方的
        "用关联程序打开" 接口 (PyInstaller 打包后也能用)。
        """
        try:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                return {"ok": False, "error": "文件不在了(可能已经移走)"}
            os.startfile(os.path.abspath(path))                 # noqa: S606
            return {"ok": True}
        except Exception as exc:                              # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # ---- AI 调色建议 ----
    def advise(self, folder, names=None, dest=None, model=None):
        """起后台线程让本机 Ollama 给这批照片出调色建议, 立即返回。

        names 为空时默认分析该文件夹里全部照片。dest 给了就先从 dest 找文件
        (照片可能已经搬过去了), 找不到再回原目录; 都找不到的会被跳过。
        model 不传则沿用界面上一次选的 (self.advice_model)。
        Ollama 连不上只会推一条错误消息, 不影响别的功能。
        """
        rows = self.state.get(folder)
        if not names:
            names = [r["file"] for r in (rows or []) if r.get("file")]
        if not names:
            return {"ok": False, "error": "这个文件夹还没分析"}
        model = (model or self.advice_model or ollama_advise.default_model())
        self.advice_model = model
        self._advice_cancel.clear()     # 新一轮开始, 清掉上一次可能留下的取消标志
        self._advice_id += 1
        aid = self._advice_id
        threading.Thread(target=self._run_advise,
                         args=(folder, list(names), dest, model, aid),
                         daemon=True).start()
        return {"ok": True, "n": len(names)}

    def cancel_advise(self):
        """请求中止当前调色建议任务。

        只置 _advice_cancel 标志: 后台线程会在解码循环 / 流式读取时停下,
        走 _run_advise 的 finally 发出 advice_done, 界面不会卡在"分析中"。
        """
        self._advice_cancel.set()
        return {"ok": True}


    def get_advice(self, folder):
        """取某个文件夹上一次的调色建议结果 (没跑过则 None)。"""
        return self.advice.get(folder)

    def get_ollama_settings(self):
        """读本机 Ollama 的本地配置 (host/port/model/max_images 等), 薄代理。

        只读本地配置文件, 不联网, Ollama 没启动也能安全调用。
        """
        return ollama_advise.get_settings()

    def save_ollama_settings(self, settings):
        """保存 Ollama 配置 (如 {"url": "127.0.0.1:11434"}), 薄代理。

        只写本地配置文件, 不联网。返回 ollama_advise.save_settings() 的结果。
        """
        return ollama_advise.save_settings(settings or {})

    def _run_advise(self, folder, names, dest, model, aid):
        """后台线程: 解码 -> 算色料统计 -> 发给 Ollama -> 推结果。

        全程不写盘, 只把图片编码进内存; 单张坏了跳过不影响整批。
        """
        try:
            st = ollama_advise.check_ollama(model)
            if not st.get("ok"):
                self.q.put(("advice_error", (folder, st.get("error", "Ollama 不可用"))))
                return

            # 解析文件路径: 优先 dest (已搬过去的), 否则回原目录
            paths = []
            for name in names:
                p = os.path.join(folder, name)
                if dest and os.path.isfile(os.path.join(dest, name)):
                    p = os.path.join(dest, name)
                if os.path.isfile(p):
                    paths.append((name, p))

            images, stats = [], []
            n_paths = len(paths)
            for i, (name, path) in enumerate(paths, 1):
                if self._advice_cancel.is_set():              # 用户点了取消, 立刻停
                    self.q.put(("advice_cancelled", folder))
                    return
                try:
                    bgr, _ = cull.decode(path)
                    s = ollama_advise.compute_color_stats(bgr)
                    s["name"] = name
                    images.append(ollama_advise.jpeg_bytes(bgr, 1024))
                    stats.append(s)
                    self.q.put(("log", f"[{i}/{n_paths}] 已准备 {name}"))
                    self.q.put(("advice_prog", {"folder": folder, "phase": "decode",
                                                "done": i, "total": n_paths, "chars": 0,
                                                "tail": ""}))
                except Exception as exc:                      # noqa: BLE001
                    self.q.put(("log", f"[{i}/{n_paths}] 跳过 {name}: {exc}"))

            if not images:
                self.q.put(("advice_error", (folder, "没有可发送的照片")))
                return

            total = len(images)
            max_images = ollama_advise.max_images()           # 每次都读配置, 改完设置不用重启
            sampled = total > max_images
            if sampled:
                idx = _even_indices(total, max_images)
                images = [images[i] for i in idx]
                stats = [stats[i] for i in idx]
            sent = len(images)

            self.q.put(("log", f"正在请求 Ollama ({model}), 共 {sent} 张…"))

            def on_progress(d):
                # 流式生成进度: 回报已产出字符数 + 最近一段原文, 界面用一行实时显示
                self.q.put(("advice_prog", {"folder": folder, "phase": "gen",
                                            "done": 0, "total": 0,
                                            "chars": d.get("chars", 0),
                                            "tail": d.get("tail", "")}))

            advice = ollama_advise.analyze_images(images, stats, model=model,
                                                  on_progress=on_progress,
                                                  cancel=self._advice_cancel)

            if aid != self._advice_id:                        # 已经是过期的任务了
                return
            payload = {"folder": folder, "model": model, "ok": True,
                       "advice": advice, "stats": stats, "sent": sent,
                       "total": total, "sampled": sampled, "gpu": st.get("gpu")}
            self.advice[folder] = payload
            self.q.put(("advice", payload))
        except Exception as exc:                              # noqa: BLE001
            if self._advice_cancel.is_set():
                # 取消导致的异常 (如 analyze_images 抛 RuntimeError("已取消")) 不算错误
                self.q.put(("advice_cancelled", folder))
            else:
                self.q.put(("advice_error", (folder, str(exc))))
        finally:
            self.q.put(("advice_done", folder))

    # ---- 模型选择 / 拉取 ----
    def list_models(self):
        """把本机 Ollama 已有的模型列出来, 供界面下拉选择 (薄代理)。"""
        return ollama_advise.list_models()

    def pull_model(self, model):
        """后台拉取一个模型, 立即返回; 已经在拉就拒绝。"""
        if not isinstance(model, str) or not model.strip():
            return {"ok": False, "error": "模型名不能为空"}
        model = model.strip()
        if self._pulling:
            return {"ok": False, "error": "正在拉取中"}
        self._pull_cancel.clear()
        self._pulling = True
        threading.Thread(target=self._run_pull, args=(model,), daemon=True).start()
        return {"ok": True, "model": model}

    def cancel_pull(self):
        """请求中止当前拉取。后台线程会在下一个进度块停下 (只丢内存, 不动已下好的部分)。"""
        self._pull_cancel.set()
        return {"ok": True}

    def _run_pull(self, model):
        """后台线程: 流式拉模型, 把进度推给界面。绝不向外抛异常。"""
        # 节流状态只活在这个线程里: 百分比涨够 1 个点 / 状态变了 / 没有总量信息时才推
        last = {"pct": None, "status": None}

        def on_progress(d):
            status = d.get("status") or ""
            total = d.get("total")
            completed = d.get("completed")
            pct = round(completed / total * 100, 1) if (total and completed) else None
            if (pct is None or status != last["status"]
                    or last["pct"] is None or pct - last["pct"] >= 1.0):
                last["pct"] = pct
                last["status"] = status
                self.q.put(("pull", {"model": model, "status": status, "pct": pct,
                                     "completed": completed, "total": total}))

        try:
            ollama_advise.pull_model(model, on_progress=on_progress,
                                     cancel=self._pull_cancel)
            self.q.put(("pull_done", model))
        except Exception as exc:                          # noqa: BLE001
            self.q.put(("pull_error", (model, str(exc))))
        finally:
            self._pulling = False

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
