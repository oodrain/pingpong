# 乒乓球触台复核工作台

这是软件工程课程第二次作业的本地试用版。项目从赛事触台检测流程延伸到专业人员的复核工作台。触台指乒乓球碰到球桌台面，工具把原视频、短球轨迹、俯视落点和复核清单放在同一个窗口里。

**当前可以试用**固定机位样片的播放、A/D 逐帧、点击落点跳帧、四角标定、确认、驳回、标疑难、补记漏报和面板调整。工作台读取合成样片的预置事件，人工记录单独保存。用户自带录像的导入、CPU 自动分析和导出属于后续集成目标。

[第二次作业说明与四人分工](docs/assignment-2.md)

## 第一次启动

准备一台 Windows x64 电脑，安装 64 位 Python 3.12，保留 Tcl/Tk 组件，并让 PowerShell 能运行 `python`。安装依赖时需要联网，之后可以离线复核样片。

1. 在仓库页面切换到 `course/video-workbench` 分支，再选择 **Code → Download ZIP**，或者用 Git 克隆该分支。解压或克隆到自己能写入的本地目录。
2. 进入 `course-workbench`，在这个目录打开 PowerShell，按顺序执行下面的命令。样片生成器会画出球桌和运动轨迹，并生成 25 fps 和 50 fps 各一段，时长为 5 秒。

   ```powershell
   python --version
   python -m venv .venv-annotation
   .\.venv-annotation\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv-annotation\Scripts\python.exe -B .\tools\generate_samples.py
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Check
   ```

3. 看到 `VIDEO_STUDIO_CHECK_OK` 后，双击 **`Start-Review.cmd`**。默认打开 50 fps 样片。以后直接双击即可。

要试 25 fps，用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Sample course25`。Git 用户也可以从仓库的上一级目录执行以下命令。

```powershell
git clone --branch course/video-workbench --single-branch https://github.com/oodrain/pingpong.git
cd pingpong\course-workbench
```

## 五分钟复核练习

1. 按空格播放。左边看原视频和短轨迹，右上看俯视球桌，右下看事件清单。球桌默认为随播放显示一个落点。
2. 将球桌切为「全部落点」，点 `f90`（50 fps）或 `f45`（25 fps）。两者都应在约 1.800 秒处同步定位原视频、俯视图和清单。按 D 前进一帧，再按 A 返回。
3. 点「标疑难」，或在其他帧点「补记漏报」并在视频上选位置。关闭重开，检查记录是否恢复。复核文件保存在 `workspace/landing_reviews/`。
4. 想验证投影方向，可点「交换 A/B」或按界面提示重标四角。视频上滚轮缩放、右键拖动，布局菜单可以隐藏、弹出或停靠面板。

## 常见启动问题

|现象|处理|
|---|---|
|找不到 `python`，或版本不同|确认安装的是 64 位 Python 3.12，并重新打开 PowerShell。|
|找不到 `.venv-annotation` 或 Python 包|在 `course-workbench` 目录重新执行创建虚拟环境和安装依赖两步。|
|提示缺少样片或结果文件|重新运行 `tools/generate_samples.py`，等两段样片都生成完成。|
|Tcl/Tk 加载失败|先用 `python -m tkinter` 检查基础安装。若测试窗口无法打开，检查安装器的 Tcl/Tk 组件。若基础窗口可打开而工作台检查失败，把 `-Check` 的报错发给 A。|
|检查通过但双击没有窗口|查看 `workspace/launcher_logs/` 下最新的 `stderr.log`，把报错和 Windows 版本发给 A。|

## 在哪里找代码和文件

- `development/video_studio/` 是工作台源码。`review_geometry/` 包含从现有项目复用的球桌平面和轨迹辅助模块。
- `tools/generate_samples.py` 生成合成视频及三份结果文件。`run.json` 记录视频指纹、尺寸和帧数，`tracks.json` 记录球轨迹，`landings.json` 记录触台候选。
- `samples/generated/` 是本地生成的演示数据，`workspace/` 是个人复核和日志。两处目录均由 Git 忽略。原赛事录像和模型权重沿用项目原有渠道。
