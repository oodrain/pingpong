# 乒乓球触台复核工作台

触台复核工作台是在 Windows 上运行的本地工具，用于核查乒乓球碰到球桌的时刻和位置。原视频、短球轨迹、俯视落点和事件清单集中在一个窗口，点击落点即可跳到对应帧。

工作台支持导入本地视频、播放、A/D 逐帧、四角标定和面板调整。加载触台事件后，点击俯视落点或清单即可跳回对应帧。复核可确认、驳回、标疑难或补记漏报，人工记录单独保存。带预置触台事件的固定机位样片可直接练习这套流程。

## 第一次启动

**完整 Windows 目录**包含 `Start-Workbench.cmd`、`runtime/` 和 `analysis_assets/`。双击 `Start-Workbench.cmd`，点击「导入视频」，工作台会自动识别同目录的 CPU 环境并开始分析。

**源码下载**需要在 Windows x64 上安装 64 位 Python 3.12（含 Tcl/Tk）。仓库的自动分析需要另备 CPU 运行文件和模型，见下文。首次安装依赖需要联网。

1. 在仓库首页选择 **Code → Download ZIP**，或者用下方的 Git 命令克隆仓库。解压或克隆到自己能写入的本地目录。
2. 进入 `course-workbench`，在这个目录打开 PowerShell，按顺序执行下面的命令。

   ```powershell
   python --version
   python -m venv .venv-annotation
   .\.venv-annotation\Scripts\python.exe -m pip install -r requirements.txt
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Workbench -Check
   ```

3. 看到 `VIDEO_STUDIO_CHECK_OK` 后，双击 **`Start-Workbench.cmd`**，再导入自己的录像。

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

源码版可以将 CPU 包解压到 `course-workbench/cpu-runtime/` 下，使 `cpu-runtime` 直接包含 `runtime/`、`analysis_assets/` 和分析程序。也可以从界面指定其他目录。可运行 CPU 目录内的 `Check CPU.cmd` 检查依赖。

1. 第一次打开工作台，点「配置 CPU」，选择解压后包含 `runtime/` 和 `analysis_assets/` 的目录。工作台会记住该路径。
2. 点「导入视频」选择自己的录像。底部开启「导入后分析」时，首次打开这段视频会自动开始分析，并在底部显示进度。已有结果的录像会直接载入结果。需要手动控制时，关闭「导入后分析」，再按需点「CPU 分析」。
3. 底部「CPU 上限」默认是 40%，可选 20%、40%、60%，设置会用于下一次分析。上限约束 Windows 后台分析进程，适用于分析阶段的解码和模型计算。较低的上限可能延长分析时间。
4. 分析完成后，轨迹和触台候选会回到工作台。可以逐条复核，或用「导出预测」导出模型的原始候选。请在清单中查看人工复核状态。

也可以通过下面的命令临时指定 CPU 运行目录。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Video "C:\Videos\clip.mp4" -AnalysisRuntime "C:\Tools\PingpongStudio_CPU_Windows_x64"
```

当前视频只显示画面时，看一下底部状态和「配置 CPU」按钮。样片使用预置结果。分析日志保存在 `workspace/analysis/jobs/`。

## 球桌标定与落点复核

1. 在固定机位录像中停在台面边线清晰的一帧。点「标定四角」，先选 A 端一个台角，沿同一长边到 B 端，再横跨 B 端到第三角，沿另一条长边返回 A 端。点击台面边线的交点。
2. 核对原画面四条实线是否贴合台面边线、黄色虚线是否对齐球网与台面的交线，并确认示意图左侧对应原片的 A 端。角点可以依次撤销。点击「确认标定」后才应用新投影。已有标定可点「检查标定」，返回保存的源帧复查。机位移动后请重新标定。
3. 俯视图用空心标记显示待复核候选，用实心标记显示已确认事件。「疑似空中」仍可从清单跳回原片检查，确认其确实触台后才显示投影。单目俯视点是对台面的投影估计，台面轮廓内的图像位置仍可能是空中的球。

## 样片复核操作

要体验合成样片，先在源码目录运行 `.\.venv-annotation\Scripts\python.exe -B .\tools\generate_samples.py`，再双击 `Start-Review.cmd` 打开 50 fps 样片。要试 25 fps，使用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-VideoStudio.ps1 -Sample course25`。

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
