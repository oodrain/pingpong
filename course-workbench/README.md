# 乒乓球触台复核工作台

触台复核工作台是在 Windows 上运行的本地工具，用于核查乒乓球碰到球桌的时刻和位置。原视频、短球轨迹、俯视落点和事件清单集中在一个窗口，点击落点即可跳到对应帧。

工作台支持导入本地视频、播放、A/D 逐帧、四角标定和面板调整。加载触台事件后，点击俯视落点或清单即可跳回对应帧。复核可确认、驳回、标疑难或补记漏报，人工记录单独保存。带预置触台事件的固定机位样片可直接练习这套流程。

## 第一次启动

准备一台 Windows x64 电脑，安装 64 位 Python 3.12，保留 Tcl/Tk 组件，并让 PowerShell 能运行 `python`。安装依赖时需要联网，之后可以离线复核样片。

1. 在仓库首页选择 **Code → Download ZIP**，或者用下方的 Git 命令克隆仓库。解压或克隆到自己能写入的本地目录。
2. 进入 `course-workbench`，在这个目录打开 PowerShell，按顺序执行下面的命令。样片生成器会画出球桌和运动轨迹，并生成 25 fps 和 50 fps 各一段，时长为 5 秒。

   ```powershell
   python --version
   python -m venv .venv-annotation
   .\.venv-annotation\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv-annotation\Scripts\python.exe -B .\tools\generate_samples.py
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Check
   ```

3. 看到 `VIDEO_STUDIO_CHECK_OK` 后，双击 **`Start-Review.cmd`**。默认打开 50 fps 样片。以后直接双击即可。

要试 25 fps，用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Sample course25`。

通过 Git 下载时，在准备存放仓库的目录执行下面的命令。

```powershell
git clone https://github.com/oodrain/pingpong.git
cd pingpong\course-workbench
```

## 播放本地视频

从窗口上方的「导入视频」选择自己的录像，仍在原视频、俯视球桌和事件清单的界面操作。也可以在 `course-workbench` 目录用下面的命令直接打开视频。空格播放，A/D 或左右方向键逐帧。固定机位可以点击「标定四角」，为当前录像设置俯视投影。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Video "C:\Videos\clip.mp4"
```

## 本地 CPU 分析

播放和人工复核只需要上面的依赖。自动分析使用单独提供的本地 CPU 运行目录，其中包含运行时和模型。检查 CPU 运行目录内的 `Check CPU.cmd` 后，用 `-AnalysisRuntime` 指定该目录。仓库源码可以独立使用，模型按项目原有渠道获取。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Video "C:\Videos\clip.mp4" -AnalysisRuntime "C:\Tools\PingpongStudio_CPU_Windows_x64"
```

打开自己的视频后点「CPU 分析」，完成后轨迹和触台候选会回到当前工作台。四角标定后，候选会同步显示在俯视球桌上。可以逐条复核，或用「导出预测」导出自动结果。分析日志保存在 `workspace/analysis/jobs/`。

## 样片复核操作

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
|Tcl/Tk 加载失败|先用 `python -m tkinter` 检查基础安装。若测试窗口无法打开，确认安装了 Tcl/Tk 组件。若只有工作台检查失败，保存 `-Check` 的报错以便排查。|
|检查通过但双击没有窗口|查看 `workspace/launcher_logs/` 下最新的 `stderr.log`，记录报错和 Windows 版本。|

## 在哪里找代码和文件

- `development/video_studio/` 是工作台源码。`review_geometry/` 包含从现有项目复用的球桌平面和轨迹辅助模块。
- `tools/generate_samples.py` 生成合成视频及三份结果文件。`run.json` 记录视频指纹、尺寸和帧数，`tracks.json` 记录球轨迹，`landings.json` 记录触台候选。
- `samples/generated/` 是本地生成的演示数据，`workspace/` 是个人复核和日志。两处目录均由 Git 忽略。原赛事录像和模型权重沿用项目原有渠道。
