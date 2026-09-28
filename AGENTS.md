# 给 Agent 的开发文档

> 这份文档写给**要修改/构建这个项目的 AI 助手**，不是给终端用户看的。
> 面向用户的说明在 `README.md`。
>
> 目标读者：拿到这个仓库、需要装环境、改代码、重新打包的 agent。
> 假设你会用命令行，但**不会**有耐性去读 1400 行的 `cull.py` 全文。

---

## 1. 这是什么

一个**本地**的照片粗筛工具：把一堆 RAW 照片按"眼睛是否合焦"排序打分，帮摄影师快速过掉跑焦的废片。

- 全离线运行，只读原片，**不会**修改原图（唯一的写操作是用户显式点"移动/复制"）
- 有两个入口：**GUI 客户端**（单窗口 WebView2）和**命令行**（`cull.py` / `run.bat`）
- GUI 跑完只写**一个缓存文件**：照片目录里的 `_cull_cache.json`（见 §7），不动原图、不生成别的文件
- 除了"眼部锐度"，还算了 **修图价值分**：锐度 / 技术画质(MUSIQ) / 姿势(yaw) / 神态(睁眼) / 美感(NIMA)
  五项加权（见 §9）。姿势/神态来自人脸关键点模型（2d106det），画质/美感走 onnxruntime
- 模型首次运行自动下到 **exe 同级的 `_internal\models\`**（`sys._MEIPASS/models`；源码运行=工具目录\models）

```
cull.py     核心算法库 + 命令行入口（无 GUI 依赖，可独立使用）
app.py      GUI 入口（pywebview 单窗口），把 cull.py 的结果渲染成网页
ui.html     GUI 界面（原生 JS，无框架、无构建步骤）
app.spec    PyInstaller 打包配置
build.py    打包脚本（调 PyInstaller 的 Python API）
run.bat     命令行启动器（拖文件夹进去即可）
```

**技术栈**：Python 3.12 + rawpy(libraw) + OpenCV(YuNet) + Pillow + pywebview/pythonnet(WebView2)

---

## 2. 环境搭建

### 2.1 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| OS | **Windows 10 / 11** | 代码里有 `os.startfile` / `explorer /select,` / `msvcrt`，无法跨平台 |
| Python | **3.12.x** | 实测 3.12.10。PyInstaller 打包的 exe 只在打包机上跑，别的机器不需要 Python |
| .NET | .NET Framework 4.6.2+ | pythonnet 用来驱动 WebView2 |
| **WebView2 Runtime** | 必需 | **Win11 自带；Win10 大概率要单独装** |

WebView2 Runtime 下载：<https://developer.microsoft.com/microsoft-edge/webview2/>

> **没有 WebView2 会怎样**：GUI 启动时弹窗报错然后退出，不会静默降级到 IE 内核
> （`app.py` 的 `main()` 里有显式检查）。命令行 `cull.py` 不受影响，照常可用。

### 2.2 建虚拟环境

```powershell
cd _cull_tool
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2.3 装依赖

```powershell
venv\Scripts\python.exe -m pip install numpy pillow opencv-python-headless rawpy
venv\Scripts\python.exe -m pip install pywebview pythonnet pyinstaller
```

**实测通过的版本**（`pip freeze` 摘出来的，别的组合没验证过）：

```
python                   3.12.10
numpy                    2.5.3
Pillow                   12.3.0
opencv-python-headless   4.14.0.94
rawpy                    0.27.1
pywebview                6.2.1
pythonnet                3.1.0
clr-loader               0.3.1   (pythonnet 带的)
bottle                   0.13.4  (pywebview 带的)
pyinstaller              6.22.3
```

### 2.4 ⚠️ opencv-python-headless 必须是 4.x

**5.x 会坏**。5.x 移除了 Haar 级联检测器，headless 包也不带级联 xml。
本项目用 YuNet（轻量 CNN，onnx）所以不受影响，但如果你看到
`opencv.data.haarcascades` 之类的报错，先查 opencv 版本。

不要装 `opencv-python`（带 GUI 的那个），装 `-headless` 版本，打包体积小很多。

### 2.5 验证环境

```powershell
venv\Scripts\python.exe -c "import cull, app; print('OK')"
venv\Scripts\python.exe -c "import webview.platforms.winforms as w; print(w.renderer)"
```

第二条**必须输出 `edgechromium`**。输出 `mshtml` 说明没装 WebView2，GUI 跑不起来。

---

## 3. 跑起来

### 3.1 GUI

```powershell
venv\Scripts\python.exe app.py
venv\Scripts\python.exe app.py "D:\照片\2026-08-21"   # 带文件夹直接启动
```

首次运行会**联网下载** YuNet 模型（230KB）到 `%TEMP%\cmdc_cull\` 或
`%ProgramData%\cmdc_cull\`。离线环境要提前把 `face_detection_yunet_2023mar.onnx`
放进那两个目录之一。

### 3.2 命令行

```powershell
run.bat                                  # 拖文件夹进窗口
run.bat "D:\照片\xxx"
venv\Scripts\python.exe cull.py "D:\照片\xxx" --open
```

---

## 4. 完整 CLI 参数

```
cull.py [folder] [选项]

folder                 照片目录; 不给则进入交互式(可拖文件夹进窗口)
--out DIR              输出目录 (默认 <照片目录>/_cull_out)
--ext a,b,c            只分析这些扩展名 (默认 RAW+图片, 逗号分隔)
--full                 全分辨率解码 (更准, 慢约 5 倍)
--jobs N               并行进程数 (默认 4; 解码受内存带宽限制, 8+ 反而更慢)
--ratio 0.55           组内锐度低于最佳 x ratio 判为"疑似模糊"
--group-ncc 0.85       相似帧分组的指纹严格度 0~1 (越大越严; 0.7~0.8 会把换了人的并进一组)
--group-window 30      连拍分组的时间窗秒数 (只有拍摄时间相隔不超过它的帧才可能同组)
--preview-size 2048    大图预览最长边; 0 = 不生成
--sheet                额外导出检测总览图 (核对人脸检测准不准)
--model PATH           指定 YuNet 模型路径 (默认自动下载)

# 挑片并移动
--keep-to DIR          把挑中的移到该目录
--keep-from FILE       从清单读文件名移动 (给了就不再重新分析)
--best-per-group       每组只留最锐的 (单张照原样保留)
--min-ratio 0.9        只留组内占比 >= 0.9
--min-sharp 15         只留眼部锐度 >= 15 (v3 起对比度归一化, 量级约 5~80)
--copy                 复制而非移动
--with-jpg             连同名 .jpg 一起搬
--dry-run              只列出会动哪些, 不动真格

# 其它
--open                 跑完自动用浏览器打开 HTML 报告
--pause                跑完等按任意键 (双击启动时用)
```

**`--keep-to` 的安全设计**（都是实测过的行为）：

- 同名 `.xmp` / `.acr` 编辑记录**一定跟随**，不然 Lightroom 的修改会和照片分家
- 目标已有同名文件**跳过，绝不覆盖**
- 目标 == 源目录直接拒绝
- 清单里不存在的文件名会被列出来，不静默忽略
- `--dry-run` 保证一个文件都不动
- 每次移动追加写 `_cull_out\移动日志.txt`

---

## 5. 代码地图（改代码前先看这里）

### `cull.py` — 算法核心，无 GUI 依赖

| 函数 | 作用 | 改动风险 |
|---|---|---|
| `RAW_EXTS` / `IMG_EXTS` | 支持的扩展名白名单 | **改这里必须同步 `decode()`**，见 §8 |
| `decode(path, full)` | 解码成 BGR ndarray。按 `IMG_EXTS` 分流到 PIL，其余走 rawpy | 高 |
| `extract_preview(path)` | 抽 RAW 内嵌预览图作兜底 | 中 |
| `analyze(path, detector, ...)` | **单张核心**：检测人脸 → 逐脸算眼睛锐度 → 取最清楚那张当主体 → 出缩略图 | **最高** |
| `measure_face(gray, ...)` | 量单张脸的眼部锐度；选脸和"手动换主体"**必须共用**同一口径 | 高 |
| `pick_subject(gray, faces, ...)` | 遍历所有脸，返回 (主体, 全部框, 全部指标) | 中 |
| `reselect_subject(row, ...)` | 手动换主体：按需重解码单张并重算 sig/锐度/缩略图 | 中 |
| `tenengrad_map` / `patch_sharpness` | Tenengrad 梯度能量 | 高 |
| `face_signature` | 人脸区域归一化灰度指纹，用于判断两帧是否同一姿势 | 中 |
| `model_dir` / `ensure_asset` / `_opencv_loadable` | 模型目录(`_internal\models`) + 按需下载 + 中文路径镜像 | 中 |
| `detect_landmarks` | insightface 2d106det：192×192 裁剪 → 106 个关键点（原图坐标） | 中 |
| `eye_metrics` | 由关键点算 (睁眼 EAR, 头朝向 yaw)：宽度比 ≈ cos(yaw)，线性映射 0~75° | 中 |
| `tech_quality` / `aesthetic_score` | MUSIQ 技术画质(0~100) / NIMA 美感(1~10)，onnxruntime | 低 |
| `_score_parts` / `edit_value_of` | 五维子分数(0~1) → 加权合成修图价值(0~100) | 中 |
| `group_frames(rows, ...)` | 把相似帧聚成组（并查集，传递闭包；候选配对受时间窗限制） | **高**（阈值/时间窗调错会误判） |
| `_same_pose(a, b, ...)` | 第 1 级判据：两帧是否"同一个人的相近姿态"，四道门槛 | **高** |
| `_burst_same(a, b, ...)` | 第 2 级判据：连拍内位置/脸宽吻合就放宽指纹门槛 | **高** |
| `_subject_may_differ(group)` | 组内主体脸 IoU 过低 → 标 ⚠ 提醒锐度不可比 | 中 |
| `_face_iou_norm(a, b)` | 主体脸框交并比（**坐标要归一化**，不同照片分辨率不同） | 中 |
| `compute_flags(rows, ...)` | **纯计算**，打上组/锐度占比/最佳/模糊/存疑标记，不碰磁盘 | 低 |
| `HardStopPool` | 能 terminate 的进程池（取消时真能停下来） | **高** |
| `write_cache` / `read_cache` | 结果缓存；字段值必须是 python 类型（见 §8.7） | 中 |
| `write_reports(...)` | 调 `compute_flags` 后写 CSV/txt/HTML | 低 |
| `write_csv(rows, path)` | 只写 CSV（GUI 导出用） | 低 |
| `list_photos(folder)` | 列出待分析文件，跳过已有 RAW 的 jpg 导出版 | 低 |
| `make_big_jpeg(path, side)` | 解码 + 缩放 + JPEG 编码，返回 bytes | 低 |
| `make_preview_jpeg(path, out)` | 写预览 jpg 文件 | 低 |
| `plan_move` / `do_move` | 移动/复制 + sidecar 跟随 + 日志 | **高**（动文件） |
| `worker_init` / `process_one` | 多进程 worker 的初始化与单张入口 | 中 |
| `read_sony_af(path)` | 读 Sony 对焦元数据（见 §10 限制） | 低 |

**分层原则**：`cull.py` 不 import `app.py`，也不 import `webview`。
多进程 worker 只 import `cull`，所以不会把 .NET 运行时拖进子进程。

### `app.py` — GUI 桥接层

一个 `Api` 类，它的公开方法**就是**网页能调的接口（`window.pywebview.api.xxx`）：

| 方法 | 网页怎么用 |
|---|---|
| `pick_folders()` / `pick_dest()` | 弹原生文件夹选择框 |
| `add_folders(dirs)` | **把文件夹加进队列**（JS 改了列表必须同步调这个） |
| `remove_folder(path)` | 队列上的 × 按钮 |
| `get_queue()` | 页面启动时拉一次（接命令行传的参数） |
| `analyze(full, jobs, force)` | 起后台线程，立即返回 |
| `cancel()` | 中止这一轮，结果全丢但文件夹留在队列 |
| `poll()` | 页面每 200ms 拉消息队列 |
| `get_batch(folder, off, limit, sort)` | 分页取卡片数据（缩略图 base64）；`sort` 决定同组是否排在一起 |
| `get_big(folder, name)` | 点开看大图，带内存 LRU |
| `get_face_choices(folder, name)` | 列出某张检出的所有脸（供"换主体"下拉，不重解码） |
| `set_subject(folder, name, idx)` | 手动换主体：重解码该张 → 重算 sig/锐度/缩略图 → 整体重算分组 |
| `move(folder, names, dest, copy, preview_dir)` | 移动/复制 + 可选生成预览 |
| `export_csv` / `export_list` | 手动导出 |
| `show_in_explorer(folder, name)` | 资源管理器里选中原文件 |
| `open_original(folder, name)` | 用系统默认看图程序打开原图（`os.startfile`） |

**关键常量**：`BATCH=60`（每次取多少张卡片）、`BIG_SIDE=1600`、
`BIG_CACHE_MAX=64`（大图 LRU，约 22MB）

**并发模型**：一个后台线程跑 `_run_queue`，所有进展通过 `queue.Queue` 传给
`poll()`。取消靠 `threading.Event`。`_run_id` 用来作废过期轮次的消息。

**排序（`Api._order_rows()`）**：界面排序下拉框三选一，决定同组照片是否挨在一起。

| `sort` | 行为 |
|---|---|
| `sharp` | 旧行为：按锐度降序平铺，不分节（默认，向后兼容） |
| `group` | 多张组集中置顶（强的组在前），组内按锐度降序，其后是单张 |
| `name` | 按文件名走，但同组聚成一块（组落在它最早那张的位置上） |

⚠ **分节模式下不能把一组切成两页**。`get_batch()` 是 offset 分页，如果一页正好
切在某组中间，组标题会在两页各出现一次、同组照片也隔着屏幕。所以碰到切中的组时，
`get_batch` 会**多带几张把整组取完**（`gs > 1` 那个 while 循环）。实测每页 3 张
也能拿全且不重复。代价是最多超发 `group_size-1` 张。

**换排序必须从头重拉**（`fSort.onchange` 调 `openFolder`）—— 顺序变了不能拿旧
的 offset 续着接。折叠状态存在 `SECS` 里，翻页时按 key 恢复。

### `ui.html` — 原生 JS，无框架无构建

改完直接刷新即可，不用重新打包（开发时）。

**JS 与 Python 的状态同步是这个项目最容易出 bug 的地方**，改之前请先读 §8。

---

## 6. 打包

```powershell
venv\Scripts\python.exe build.py
```

产物在 **`../_cull_dist/选图工具/`**（仓库外），约 180MB。

> ⚠️ **只能拷整个文件夹，不能只拷 exe**。依赖全在 `_internal/` 里。

`build.py` 用的是 PyInstaller 的 Python API 而不是命令行，目的是绕开
cmd.exe 管道的编码问题（控制台代码页会把中文路径搞坏）。

### app.spec 里几个非默认设置

```python
collect_all('webview')      # webview/js/ + WebView2 的 .NET dll
collect_all('pythonnet')    # .NET 运行时 dll
excludes=[matplotlib, scipy, pandas, PyQt5, PySide6, IPython, ...]
```

**`collect_all('webview')` 不能省**，原因见 §8。

### 排查清单

| 症状 | 大概率原因 |
|---|---|
| 界面全白/空白 | `_internal\webview\js\*.js` 没进包 → 查 `collect_all('webview')` |
| `Failed to load Python DLL` | 双击了 PyInstaller 中间目录里的空壳 exe → 只认 `_cull_dist` 那份 |
| 启动时弹 WebView2 缺失 | 目标机器没装 WebView2 Runtime |
| 界面显示但点了没反应 | 见 §8 的 JS/Python 状态同步问题 |
| 打包报 `PermissionError: ClrLoader.dll` | 上一次的 exe 还在跑，先关掉 |

---

## 7. 输出行为（重要）

**GUI 客户端：只在照片目录里写一个结果缓存文件 `_cull_cache.json`，不生成别的。**

结果只在窗口里显示；大图在内存 LRU 里。想落盘得点「导出 CSV…」/「导出勾选清单…」，
或者勾上「生成预览 jpg」并指定目录。

**结果缓存**（`cull.py` 的 `write_cache` / `read_cache`）：

- 位置：**照片目录里的 `_cull_cache.json`**（`cull.py` 的 `CACHE_NAME`；以前在
  `%LOCALAPPDATA%`，现已改到照片文件夹里，用户要求）
- 目的：同一批照片第二次点「开始分析」直接读结果，不再逐张解码 RAW
- 勾选框「忽略缓存重扫」= `analyze(full, jobs, force=True)`，用完自动取消勾选
- 过期自动重扫：`version` / `full` 标志 / 照片集合 / 每张的大小+修改时间，任一对不上就重算
- **不能改用 CSV 当中转**：CSV 写的是已格式化的字符串（`"1/125"`、`"眼睛"`），
  而 `get_batch()` 要的是原始类型（`(1,125)` 元组、`"face"`）
- **必须存 `sig`**：`group_frames()` 靠它做 NCC 相似度分组，少了它所有照片会退化成
  单张组，"组内最佳/疑似模糊"全失效。所以缓存存的是**原始测量值**，
  `group`/`soft`/`best_in_group` 交给 `read_cache` 里的 `compute_flags()` 重算
- **必须存 `_face_box` / `_face_metrics` / `subject_idx`**：前两个给 ⚠ 标记和
  「换主体」下拉用，后者记着当前选的是第几张脸
- 体积：实测约 8.4KB/张（无脸时），54 张一批约 0.4MB
- 命令行**不写**缓存

**命令行：写到 `<照片目录>/_cull_out/`**

| 文件 | 内容 |
|---|---|
| `选图报告.html` | 缩略图对比页，可筛选/勾选 |
| `选图报告.csv` | 每张的分数 + EXIF + AF 设置 |
| `各组最佳.txt` | 相似帧分组明细 |
| `疑似模糊.txt` | 组内被判定偏软的那些 |
| `preview/` | 2048px 大图 JPEG |
| `移动日志.txt` | 每次移动/复制追加记录 |

---

## 8. 已知的坑（改代码前必读）

这几条都是**实测踩出来的**，不是理论推测。

### 8.1 轮询必须用递归 setTimeout，不能用 setInterval

```js
// ❌ 会卡死: 界面永远停在"正在分析", 进度不动, 取消也没反应
setInterval(async function(){ var msgs = await api.poll(); handleMsgs(msgs); }, 200);

// ✅ 正确: 后一次一定等前一次回来才排下一次
async function pollOnce(){
  try { var msgs = await api.poll(); if (msgs) handleMsgs(msgs); }
  catch(e){}
  finally { setTimeout(pollOnce, 200); }
}
```

`setInterval` 不等上一次 `await` 返回就发下一次。pywebview 的 JS 侧用 `valueId`
存回调，两个 `poll()` 撞在一起会**互相覆盖，消息进了错的任务**。

`handleMsgs` 外面必须包 `try/finally` —— 单条消息处理出错也不能让轮询停摆。

### 8.2 JS 改了列表必须同步调 Python

```js
// ❌ 灾难: Python 那边队列是空的, analyze() 返回"先选一个照片文件夹",
//    但界面已经把文件夹标成"分析中" -> 永远卡住, 连取消都点不到
$('bPick').onclick = async function(){
  var dirs = await api.pick_folders();
  dirs.forEach(function(d){ FOLDERS.push({path: d, state: 'wait'}); });
  drawFolders();                      // 只画了界面
};

// ✅
await api.add_folders(dirs);           // 这行不能少
```

同理，队列上的 × 必须调 `api.remove_folder()`，否则 Python 还留着路径，
再加回来会被 `add_folders` 当成"已存在"跳过 —— **那个文件夹再也分析不了**。

### 8.3 重新分析"已完成"的文件夹，targets 必须含 `done_dirs`

第一遍跑完，文件夹会从 `queue_dirs` 挪进 `done_dirs`。而界面 `updateRun()`
只排除 `state==='run'`，所以"已完成"的文件夹**可以**再点「开始分析」。

于是 `analyze()` 最初只读 `queue_dirs`，第二遍就返回 `{"ok":false}`，
后台线程压根没起，界面永远停在"正在分析…"、取消按钮也点不动。
现在 `analyze()` 算的是 `queue_dirs + done_dirs`。

配套两个坑：

1. **不能重复 append `done_dirs`** —— 重跑时它已经在里面了，再 append 一次
   就重复，第二遍 targets 里同一路径出现两次。循环开头要先 remove。
2. **不能靠"不在 queue_dirs 里"反推是否已完成** —— 循环开头就把 d 摘出队列了，
   所以被取消的那个同样不在队列里，会被误判成已完成而**从界面上彻底消失**。
   必须用显式的 `handled` 集合记下"这轮真的处理完了哪些"。

另外 `analyze()` 出错是 **resolve `{ok:false}` 而不是 reject**，所以 ui.html 的
`.then(function(r){...})` 里必须把 `state` 退回 `wait`，否则全卡在 `run`，
`bRun` 永久禁用（这正是"提示正在分析…且无法取消"的表现）。

### 8.4 取消只能置标志，不能动 `_run_id`

```python
def cancel(self):
    self._cancel.set()        # ✅
    # self._run_id += 1      # ❌ 千万别
```

递增 `_run_id` 会让后台线程以为任务过期**直接退出**，于是不发 `idle`，
界面永远卡在"分析中"。`_run_id` 只在 `analyze()` 开始新一轮时递增。

### 8.5 光置标志停不住 worker，必须 terminate

`shutdown(wait=False, cancel_futures=True)` **停不下**已经在跑的任务：它只对
*还没开始* 的 future 生效，而 `ProcessPoolExecutor` 一旦把任务全提交，队列里
就没有"还没开始"的了。实测点了取消之后**又解码了 48 张**，4 个 worker 全程占着
几百 MB 内存继续读盘，界面却已经显示"已取消"。

所以取消要走 `HardStopPool.stop()` → `terminate()` 掉所有 worker（正在解码的
那张当场掐断，只丢内存里的结果，不动源文件）。

两个坑：

1. **别把 `shutdown()` 写在 `with` 块里靠 `__exit__` 兜底** —— `__exit__` 走的是
   `shutdown(wait=True)`，会把所有 future **drain 完**。
2. **terminate 之后池子会变 broken**，在途 future 抛的是 `BrokenProcessPool`
   而不是 `CancelledError`。直接冒泡会在界面上弹一屏 traceback，所以
   `_analyze_one` 里要用 `_cancelled(run_id)` 判断该报什么。

另外 `cancel()` 是从 JS 桥线程调的，和后台线程 drain 有竞态，所以循环内也留了
`if self._cancelled(run_id): break` —— 哪条路径先到都能立刻停。

### 8.6 预览 jpg 必须在搬文件**之前**生成

勾了"移动"的话，搬完源文件就没了，再生成就找不到源。
`Api.move()` 里顺序是 `_write_previews()` → `do_move()`。

### 8.7 numpy 标量会让缓存静默失效

`np.float32` **不能** `json.dumps`。而 `face_cx` / `face_cy` / `face_area_pct`
都是 numpy 算出来的，直接丢进 `write_cache()` 会抛
`TypeError: Object of type float32 is not JSON serializable`。

危险的是 `write_cache` 外面包了 `try/except` 且不抛出，于是**每张照片都
写不进去、而程序毫无察觉** —— 缓存功能看起来"配了"但从来没生效过。

所以 `analyze()` 里把选中的脸/眼框显式转成 python `float`，`write_cache`
里再用 `_py()` 兜底（`np.generic` → `.item()`，`ndarray` → `.tolist()`，
递归处理 list）。**新增任何来自 numpy 的字段时记得走 `_py()`。**

### 8.8 `_face_box` 必须进缓存

`_subject_may_differ()` 靠 `_face_box` 算主体脸框的 IoU 来打 ⚠。缓存里
漏了它，第二次读缓存就再也标不出"主体可能不是同一个人"。

`_CACHE_FIELDS` 里已经包含 `_face_box` / `_eye_boxes` / `_face_boxes`。

### 8.9 不能抽 RAW 内嵌预览图当预览

看着很诱人：0.000s vs 0.30s，快 300 倍。但实测 A7R III 的内嵌预览
**1616x1080 却转了 90 度**，而且**没有 EXIF Orientation 标记**，程序无法自动转正。
拿侧躺的人像给 AI 看，建议会完全离谱。所以老老实实重新解码。

### 8.10 `IMG_EXTS` 和 `decode()` 必须同步

加了新后缀但 `decode()` 没认，它会掉进 rawpy 分支，然后报"找不到内嵌预览图"
（那是给 RAW 用的兜底，对普通图片没意义）。`decode()` 现在按 `IMG_EXTS` 判断。

### 8.11 `collect_all('webview')` 不能省

`webview/js/` 用 `os.path.realpath(__file__)` 读，PyInstaller 默认把纯 Python
塞进 `base_library.zip`，`open()` 读不到 → 界面直接白屏。
pythonnet 同理（.NET dll 必须落在磁盘上）。

### 8.12 多进程 + 打包必须 `freeze_support()`

Windows 用 spawn 启动子进程。`app.py` 和 `cull.py` 的 `__main__` 里都有，
**别删**。测试脚本如果直接调 `_analyze_one()` 也得有 `if __name__ == "__main__"` 保护。

### 8.13 OpenCV 线程超订

OpenCV 默认开满所有核。4 个进程 × 20 线程 = 80 线程抢 20 核，
实测把 54 张的处理从 13 秒拖到 **44 秒**。所以多进程时每个进程的
OpenCV 线程数被压到 1（`worker_init` 的 `cv_threads` 参数）。

### 8.14 run.bat 故意只用 ASCII

cmd.exe 读 .bat 时按控制台代码页解析，里面写中文会被拆成乱命令（实测过）。
中文提示全放在 `cull.py` 里，由 `chcp 65001` 保证显示。**改这个 bat 请保持纯英文。**

### 8.15 跨盘移动

`os.path.commonpath` 在 `F:\` 和 `C:\` 之间会抛 `ValueError`。
`do_move()` 先比盘符再算公共路径。搬到别的盘是支持的。

---

## 9. 算法与性能

### 评分逻辑

1. `rawpy` 解码 RAW，拿原始像素（默认半分辨率，约 0.3s/张），**多进程并行**
2. OpenCV **YuNet** 轻量 CNN 检测人脸，给出双眼等 5 个关键点（**返回全部人脸**）
3. 逐张脸在眼睛位置的原始像素上算 **Tenengrad 梯度能量 ÷ 区域对比度**，
   **取最清楚的那张脸当主体**
4. 把**拍摄时间相近**（时间窗内）且「脸的位置 + 脸大小 + 焦距 + 脸部指纹 NCC」
   相符的帧聚成一组，**只在组内比较**（不同距离/焦段的锐度本来就没有可比性）
5. 组内挑最锐的，低于组内最佳 × ratio 的标为"疑似模糊"

### 多脸怎么选主体（`analyze()`）

**取所有脸里眼部锐度最高的那张**，不是最大的那张。实测一批 25 张合影，
`max(面积)` 选出的脸有 **21/25** 不是最清楚的，偏差 15%~97% ——
合影里主角的脸往往不是最大的（大的在前排路人）。

`sig`（人脸指纹）必须取**选中那张脸**的，否则分组会拿 A 图的第 3 张脸去和
B 图的第 1 张脸比，主体就漂了。

**怎么量的锐度**（`measure_face()`，v4 起三重归一化）：

1. **尺度**：眼框边长按**瞳距**取（`0.70×瞳距`），重采样到固定 `PATCH_OUT=96`px →
   消除"远景小脸 vs 近景大脸"的像素数差异
2. **朝向**：按 `1/cos(yaw)` 横向多取样（yaw 来自 `eye_metrics`）→ 抵消侧脸时远侧眼
   被压缩，修掉"3/4 侧脸明明合焦却被评低分"
3. **对比度**：`Tenengrad 均值 ÷ crop.std()`（+1.0 防除零）→ 抵消阴影/欠曝（v3）

两只眼取平均。注意分数量级随 v4 再次变化（典型 ~5~80）。

**为什么要除以对比度（v3）**：Tenengrad 正比于局部对比度，同一姿势只要脸上
暗一点分数就掉一大截，会被误判成"疑似模糊"。实测这批 1349 张，
`corr(log 眼部锐度, 亮度) = 0.33`，而且**组内 soft 帧一律比组内最佳暗
30~53 个亮度单位**。除以 `crop.std()` 后同一姿势不同曝光的分差基本消掉
（例：组113 的阴影帧组内占比从 0.35~0.39 抬到 0.58~0.67），而真正的跑焦
（高频边缘被抹平、对比度还在）依然掉分。
⚠ 用 `÷std²` 会**过校正**（阴影帧反而超过亮帧），`÷std` 才对。

⚠ **这个度量仍有的偏差**：

- Tenengrad 是**均值**，会被**脸的大小**带偏。实测同一张图里 w=112 的远处小脸
  拿 8297 分，w=265 的近景大脸只拿 1134 分（归一化之前的数据）。所以选脸时
  "最清楚"未必等于"最近/最大"，这也是把它做成手动可控的原因。
- 早期试过用"对比度归一化"去改**选主体**，跟"面积最大（最近）"的**一致率都只有
  20%~32%**，当时没采用 —— 那是在比"不同的脸"；而除法真正解决的是**同一张脸
  不同曝光**，用在组内比软硬上才对症。
- 归一化**改变了分数量级**：典型眼部锐度从几百~几千变成 **~5~80**。
  `--min-sharp` 这类绝对值阈值要跟着改（默认 0，不受影响）。
- 25 张里有 **6 张**头部只领先次头部 1.01~1.09 倍 —— 这两张脸人眼也很难分高下。

**所以做成手动可控**：多脸卡片上给「主体」下拉，列出每张脸的锐度/脸宽/位置，
`set_subject()` 换完会重算 sig 和分组。换主体要**重新解码那一张**（sig 和
`face_sharp` 都依赖原图像素，缓存里算不出来），约 0.3s。

**⚠ 已知局限**：即使程序选了"最清楚"那张，主体仍可能**在人之间跳**。所以
`_subject_may_differ()` 会算出组内主体脸框的 IoU，低于 0.30 就在卡片上标 ⚠，
**提醒用户这种组的组内锐度不可比**。要根治得做跨照片的主体追踪，目前没做。

### 分组（`group_frames()`）

**并查集**（传递闭包），但候选配对被**时间窗**限制：只有拍摄时间
（EXIF `datetime_original`）相隔不超过 `BURST_SECS`（默认 30 秒）的帧才两两比对。
这道限制是必需的 —— 纯全局闭包会滚雪球：实测一批 1349 张，全局闭包把
**2 小时里 254 张互不相关的照片**并成一组（主体脸中心横跨 0.34~0.68）。
加了时间窗后最大组降到 **51 张 / 39 秒**，而真正的一串走位连拍
（05095~05145，14 秒 25 张）能收进一组。

每对被时间窗允许的帧，判定分两级：

1. `_same_pose()` 常规判据：`pos_tol` 0.12 / `size_tol` 0.30 / `focal_tol` 0.15 /
   指纹 `ncc_min` 0.85，四道全过
2. `_burst_same()` 连拍判据：`BURST_POS_TOL` 0.06 / `BURST_SIZE_TOL` 0.20 /
   指纹 `BURST_NCC` 0.20 —— 时间很近时只要位置/脸宽几乎没变就放宽指纹门槛

**为什么要第 2 级**：人走动/转头会让整脸指纹掉得非常低。实测同一串走位连拍
里相邻帧的 NCC 能掉到 **0.05**，只靠 0.85 的门槛会把这一串拆成 5 组。
第 2 级靠"时间近 + 位置/脸宽吻合"把连拍收回一组，指纹下限 0.20 仍用来
挡住画面里换人/换背景的情况。

**没有 EXIF 时间的帧**退回按文件顺序相邻（`|i-j| <= 3`），只走第 1 级判据。

**`ncc_min` 为什么是 0.85**（另一批 25 张会议照实测的分界，现在是主判据之一）：

- 原来的 0.80：300 个组合里**一对都过不了**（最高才 0.794）→ 25 张全是单张组
- 调到 0.60：分出 3 个组，但人工核对发现**其中 2 个是换了人**
- 逐对量：**正确的对 NCC=0.905，错误的对是 0.847 和 0.633**。位置差、脸宽差、
  IoU 全都分不开（正确对的位置差 0.085 反而比错误对还大）。**只有 NCC 能干净切开**：
  ≥0.85 时恰好只留下正确的那一对，≥0.92 就什么都不剩

⚠ **0.85 是拟合那 25 张得出的，样本只有 3 对**，而且那批数据每对都是
"站着几乎不动"，所以它只是这个场景的分界；有走位/转头的连拍靠第 2 级兜底。
换场景时用 `--group-ncc`（指纹门槛）和 `--group-window`（时间窗秒数）现场调。

`pos_tol` 是区分「不同的人」最有效的一道闸。`_same_pose()` 里
还预留了 `iou_min`（主体脸框重叠下限，默认 0），需要时可以开。

### 性能（实测，20 核机器，54 张 42MP ARW）

| 配置 | 总耗时 |
|---|---|
| 1 进程 | 26.0 s |
| 2 进程 | 19.2 s |
| **4 进程（默认）** | **12.7 s** |
| 8 进程 | 13.3 s（反而更慢） |

只到 2 倍加速是因为瓶颈在 **RAW 解码的内存带宽**，不是磁盘
（F: 实测 2560 MB/s，86MB 一张只要 34ms）。加更多进程没收益。

单张时间构成：解码 317ms / 缩略图 / 人脸检测 80ms / 读文件 32ms / **眼睛锐度 1ms**。

### 准确度的边界（必须如实告知用户）

- 眼睛区域的梯度能量会被**框景大小**影响：近景特写脸上大片平滑皮肤，分数天然
  偏低；中远景塞满头发睫毛草叶，分数天然偏高。**跨照片的绝对值不能直接比清晰度。**
  光线/阴影已由"÷ 区域对比度"(v3) 抵消掉大部分，但框景大小这一项归一化不掉。
- 只有**同一组内**比较才可靠。这也是"疑似模糊"只在组内判定的原因。
- 有误报漏报，**最终以肉眼复核为准**。
- 眨眼/表情/姿态不判断（那需要 Aftershoot / Imagen 这类商业模型的活）。

---

## 10. 硬件/相机的已知限制

### Sony AF 元数据大部分拿不到

脚本读三个**未加密**的 maker note 标签：`0x201b`(FocusMode)、
`0x201c`(AFAreaModeSetting)、`0x201d`(FlexibleSpotPosition 坐标)。

但**实测这批 A7R III + 原厂 FE 镜头照片**：121 张里绝大多数是 `AF-A + Wide`，
坐标全是 (0,0)。Wide 模式下相机自己选点，而 **ILCE 机型不记录它实际选了哪个点**
（那段在加密的 0x94xx 块里，连 ExifTool 都没解开）。
只有用 Center 或 Flexible Spot 拍摄才拿得到真实坐标。

> **所以：想要"相机到底对在脸上还是背景上"这个判断，必须改用 Center 或
> Flexible Spot 拍摄。** Wide 拍的话这条路是堵死的，不是代码的问题。

---

## 11. 开发时的验证手段

**GUI 的 JS 可以直接用 WebView2 的调试端口驱动，比用像素坐标点可靠得多：**

```python
import os, subprocess, json, time, urllib.request
PORT = 9222
env = dict(os.environ,
           WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=f"--remote-debugging-port={PORT}")
p = subprocess.Popen('"选图工具.exe" "D:\\照片\\xxx"', env=env)
# 轮询 http://127.0.0.1:9222/json 拿 webSocketDebuggerUrl,
# 用 CDP 的 Runtime.evaluate 就能读/改页面状态, 还能 Page.captureScreenshot
```

- 跑**未打包**的源码版时可以直接 `webview.settings['REMOTE_DEBUGGING_PORT'] = 9222`
- **注意**：手工调 `api.poll()` 会和页面自己的轮询**抢消息**，
  导致页面收不到。诊断时只读 DOM，别去调 `poll()`。
- PyInstaller 的中间产物会留在 workpath，**只有一个 exe 没有 `_internal`**，
  双击必报 `Failed to load Python DLL`。`build.py` 已把中间目录挪到
  `%TEMP%\cmdc_cull_build\` 并在每次打包前清理。

---

## 12. 提交前检查

```powershell
# 1. 代码能跑
venv\Scripts\python.exe -c "import cull, app; print('OK')"

# 2. 命令行路径没坏 (CSV 应与改动前逐字节一致)
venv\Scripts\python.exe cull.py "D:\测试照片" --out "$env:TEMP\culltest" --preview-size 0

# 3. 打包 + 确认资源进包
venv\Scripts\python.exe build.py
Test-Path ..\_cull_dist\选图工具\_internal\webview\js\api.js
Test-Path ..\_cull_dist\选图工具\_internal\ui.html
```

**不要提交**：`venv/`、`_cull_dist/`、`_build/`、`_cull_out/`、`__pycache__/`
（`.gitignore` 已覆盖，但改动后请确认 `git status` 干净）。

**别把 `build.py` / `app.spec` 里的路径改回硬编码绝对路径** —— 用
`os.path.dirname(os.path.abspath(__file__))` 推导。
