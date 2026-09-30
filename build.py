"""用 PyInstaller 的 Python API 构建, 避开命令行管道的中文编码问题。"""
import os
import shutil
import sys
import traceback

TOOL = os.path.dirname(os.path.abspath(__file__))
# 成品放在仓库外的 ../_cull_dist, 免得跟源码混在一起
DIST = os.path.abspath(os.path.join(TOOL, os.pardir, '_cull_dist'))
# 中间产物放 TEMP, 不要放在 _cull_tool 下面: PyInstaller 会在 workpath 里留一个
# "只有 exe、没有 _internal" 的空壳, 尺寸跟真 exe 几乎一样, 很容易被误当成成品双击,
# 然后报 "Failed to load Python DLL" (因为找不到 _internal\python312.dll)。
WORK = os.path.join(os.environ.get("TEMP", TOOL), "cmdc_cull_build")

log = open(os.path.join(TOOL, '_build.log'), 'w', encoding='utf-8', buffering=1)


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + '\n')


def main():
    os.chdir(TOOL)
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    if not os.path.isdir(DIST):
        os.makedirs(DIST)
    # 上一轮可能留下的空壳 exe 先清掉
    stale = os.path.join(TOOL, '_build')
    if os.path.isdir(stale):
        shutil.rmtree(stale, ignore_errors=True)
        p("removed stale workdir:", stale)
    p("workdir:", os.getcwd())
    p("dist:", DIST)
    p("build work:", WORK)

    from PyInstaller.__main__ import run
    args = [
        'app.py',
        '--noconfirm', '--clean',
        '--name', '选图工具',
        '--noconsole',
        '--distpath', DIST,
        '--workpath', WORK,
        '--specpath', WORK,
        '--add-data', f'{TOOL}\\ui.html;.',
        # webview 的 js/ + WebView2 的 .NET 封装 dll、pythonnet 的运行时 dll
        # 都必须原样落在磁盘上, 混进 base_library.zip 会白屏
        '--collect-all', 'webview',
        '--collect-all', 'pythonnet',
        # onnxruntime 的 capi dll / 各 provider 必须落在磁盘上, 不能只靠 hidden-import
        '--collect-all', 'onnxruntime',
        '--hidden-import', 'onnxruntime',
        '--hidden-import', 'rawpy',
        '--hidden-import', 'clr',
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'scipy',
        '--exclude-module', 'pandas',
        '--exclude-module', 'IPython',
        '--upx-dir', '',
    ]
    # 把项目里已下好的模型一并打进 _internal\models。打包后 cull.model_dir() 返回
    # _MEIPASS\models (= _internal\models), ensure_asset() 发现文件已在就不会再联网下载。
    # models\ 不进版本库 (见 .gitignore), 全新 clone 上没有 -> 跳过, 装出来仍会在首次
    # 运行按需下载。注意: 里面可能含 1.7GB 的 aes_v25, 会明显拖慢打包并撑大成品。
    models_dir = os.path.join(TOOL, 'models')
    if os.path.isdir(models_dir) and os.listdir(models_dir):
        args += ['--add-data', f'{models_dir};models']
        p("bundling models from", models_dir)
    else:
        p("no local models/ dir -> exe will download models on first run")
    p("running PyInstaller with", len(args), "args")
    try:
        run(args)
    except SystemExit as e:
        p("SystemExit:", e.code)
    except Exception:
        p("EXCEPTION:\n" + traceback.format_exc())
        return 1
    p("--- dist 内容 ---")
    for root, dirs, files in os.walk(DIST):
        for f in files:
            fp = os.path.join(root, f)
            p(f"  {os.path.relpath(fp, DIST)}  {os.path.getsize(fp)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
