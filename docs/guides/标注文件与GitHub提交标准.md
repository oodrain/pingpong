# 标注文件与GitHub提交标准

仓库按初标来源成员归档复核，让同一视频的initial和review相邻。例如A复核D时，文件放labels/D/视频/review，metadata和每条记录的annotator仍是A，review_of.annotator仍是D。按复核者存放的交换包也兼容。初标目录必须匹配初标者；复核目录必须匹配实际复核者或绑定的初标者，不能改写真实身份来适配目录。使用仓库自带的新版校验器：python tools/validate_labels.py validate labels。

版本：2，2026-10-07。适用于四人小组内部补标；校验器继续接受v1，新本地窗口输出v2，旧CLI默认输出v1。

大家可以使用现有工具，也可以自己做工具。统一的是数据版本、判断口径和交回的文件。先每人完成一段，B检查格式和帧号，再批量推进。

## 1. 每个人要做什么

1. 由组长在工具外分配，通过文件传输领取视频和对应对象JSON，保持视频编号不变；文件可放任意目录。
2. 完整查看视频，记录可判断的触台事件；看不清的记录疑难区间。
3. 在窗口导出当前或批量结果，自动校验；其他工具按下面格式交付并运行检查脚本。
4. 把自己的结果提交到GitHub分支，由B检查后交A合并。
5. 初标和复核分别提交。复核由另一名成员完成，原初标保留。

人员仍用A/B/C/D，分配及复核安排由组长组织。既有任务表可作参考，本地导入和默认格式校验不强制任务分配。复核者必须不同于初标者。

## 2. 统一判断口径

- **目标**：触台帧号及该帧的可见球心，使用原图像素坐标。这里的x/y不是球台上的厘米坐标。
- **帧号**：顺序解码编号，第一帧为0。按帧查看时不能把播放器时间或检测器处理帧当作标注帧号。
- **坐标**：左上角为(0,0)，x向右、y向下；缩放或放大查看后换算回原图。
- **判断依据**：看疑似接触前后连续画面，通常至少前后3帧；需要时扩大范围。球框和轨迹用于参考，不能直接代替触台判断。
- **看不清**：遮挡、接触发生在两帧之间或球心无法定位，记录区间和原因，保留精确帧号及坐标的空缺。
- **干扰**：确定触网、触地、击球时，可另存干扰事件，不放入触台文件。
- **完成**：本地模式由成员直接声明已完整人工检查，不要求程序显示过全部帧；播放过不自动完成。实际显示范围与人工声明分别记录。
- **使用限制**：没有事件记录不自动等于负样本；独立复核一致也不保证标签正确。初标、复核及疑难分别保留。

第一次使用自己的工具时，要核对首尾帧、实际帧数和尺寸；用原图与缩放视图点选同一点，坐标恢复误差不超过1像素。保存后重启，记录应一致。输入与声明信息不符时先停止标注，交B核对。

## 3. GitHub里放什么

仓库中的正式交换目录统一叫labels，按成员、视频、轮次组织：

```text
labels/
  README.md
  A/
    03_007/
      initial/
        metadata.json
        events.jsonl
        uncertain.jsonl
        non_table_events.jsonl   # 可选
  B/
    03_007/
      review/
        metadata.json
        events.jsonl
        uncertain.jsonl
        non_table_events.jsonl   # 可选
```

每段、每轮必交metadata.json、events.jsonl、uncertain.jsonl。没有对应记录时JSONL保留为空文件，不能省略。non_table_events.jsonl可选；需要解释修改时可加简短README.md。

JSON和JSONL均为UTF-8编码，不带BOM。JSONL每行一个JSON对象，不使用外围数组、注释、尾逗号、NaN或Infinity。坐标保存为数值，帧号保存为整数，布尔值使用true/false。

换行建议LF，仓库.gitattributes也会统一labels文本换行。复核来源的三个交换文件哈希按CRLF转LF后的字节计算，由reference命令完成，避免Windows与Git换行转换造成假分歧。视频和原始对象JSON的输入哈希按原始文件字节计算。

以下示例只解释格式，不能复制成自己的标注答案。

## 4. 触台文件events.jsonl

```json
{"event_id":"03_007_A_initial_000060","video_id":"03_007","frame_id":60,"x":787.6,"y":550.4,"event_type":"table_bounce","review_state":"provisional","round":"initial","annotator":"A"}
```

|字段|要求|
|---|---|
|event_id|非空字符串，在这一份提交内唯一；修改同一事件时保留编号|
|video_id|统一编号，如03_007，与任务清单和目录一致|
|frame_id|整数，0 ≤ frame_id < 实际帧数|
|x、y|原图像素数值，可带小数；0 ≤ x < 宽，0 ≤ y < 高|
|event_type|固定table_bounce|
|review_state|固定provisional，表示人工候选；复核也先保留此状态|
|round|initial或review，与所在目录一致|
|annotator|实际标注者A/B/C/D，与metadata和任务分配一致；复核可在初标来源成员目录归档|

按frame_id升序排列。同一视频、成员和轮次，一帧最多一条触台记录，修改坐标应更新原记录。不同成员和轮次可以使用相同event_id，汇总时以成员、视频、轮次、event_id联合识别。

## 5. 疑难文件uncertain.jsonl

```json
{"event_id":"03_007_A_initial_uncertain_000032_000039","video_id":"03_007","frame_start":32,"frame_end":39,"reason":"小球被遮挡，无法确定触台时刻","review_state":"uncertain","round":"initial","annotator":"A"}
```

- frame_start、frame_end是整数，两端都包含，满足0 ≤ 起始 ≤ 结束 < 实际帧数。
- reason必须写清原因，例如遮挡、帧间接触、球心不可见、触网与触台难分。不要只写“不确定”。
- 状态固定uncertain，其他身份字段与metadata一致。
- 不添加猜测的frame_id、x、y；按起止帧排序，完全相同区间合并为一条说明。

疑难区间和触台记录可能在时间上重叠；这是需要人工讨论的证据，不让程序直接删除或当作负例。

## 6. 来源与完成情况metadata.json

```json
{
  "schema_version": 2,
  "video_id": "03_007",
  "annotator": "A",
  "round": "initial",
  "width": 1920,
  "height": 1080,
  "n_frames": 117,
  "fps": 25.0,
  "video_sha256": "88859db2a0b95be5fcdfc5e047cdee84ffe9b2f917efea3cb9a45743a35fa8fb",
  "annotation_sha256": "0eb1838c6d4c7702bf0e8139a35f0f97f688719be2e9e3642cf4dce78cbc6f16",
  "completed": false,
  "completion_basis": null,
  "displayed_ranges": [[0, 60]],
  "checked_ranges": [[0, 60]],
  "no_events_reason": "",
  "label_version": 1,
  "updated_at": "2026-10-07T20:00:00+08:00",
  "tool": "my-annotation-tool/v2",
  "source_type": "manual",
  "review_of": null
}
```

- width、height、n_frames、fps来自实际解码；每段分别填写，不照抄117或1920×1080。
- 两个SHA256是输入视频及原始对象JSON的文件哈希，用小写64位十六进制保存。文件搬家后哈希相同，仍能确认来源。
- completed表示本轮是否人工完整检查。完成后checked_ranges必须为[[0,n_frames-1]]；未完成则填实际检查范围，区间两端包含，排序并合并相邻/重叠区间。不需要提交每帧按键记录。
- v2的completion_basis：完成时固定manual_attestation，未完成时为null，表示人工声明，不代表程序核验了全帧查看。
- v2的displayed_ranges：工具实际显示帧的范围，两端包含、排序且合并；允许空或局部范围，完成也不要求全覆盖。旧标准导入缺少显示记录时为null，不能编造覆盖。
- checked_ranges表达人工检查声明。完成确认即声明整段，草稿保留明确记录的检查范围，不能混用displayed_ranges。v1保留原完成规则，不要求新增字段。
- 完整检查后没有可确认触台，events为空，并在no_events_reason说明原因。它仍不代表已取得可靠的全部负样本。
- label_version从1开始，已交版本修改时加1；Git保留旧内容。updated_at带时区，更新时填写。
- source_type为manual或imported。接管旧标签、导入试点等情况用imported，加note说明来源，不能宣称自己重新完成了判断。
- 初标review_of为null。复核review_of记录所依据的初标成员、label_version，以及metadata.json、events.jsonl、uncertain.jsonl三文件的SHA256，字段分别为annotator、label_version、metadata_sha256、events_sha256、uncertain_sha256。可用下面的reference命令自动生成，避免手填。

自制工具应在导出时计算输入哈希。手工核对可用PowerShell：

```powershell
(Get-FileHash -LiteralPath "视频文件.mp4" -Algorithm SHA256).Hash.ToLower()
```

## 7. 可选干扰文件

```json
{"event_id":"03_007_A_initial_net_000095","video_id":"03_007","frame_id":95,"x":null,"y":null,"event_type":"net_contact","reason":"连续画面可见球撞网","review_state":"provisional","round":"initial","annotator":"A"}
```

event_type只用net_contact、floor_contact、racket_contact、other。必须有reason，坐标可同时为null；填写时遵循原图范围。按frame_id、event_type排序，同帧同类型不重复。不要自行映射成公开GT的valid=false。

## 8. 导出与检查

检查脚本是 [标注工具 ZIP](../../tools/pingpong-annotation-local-20261007.zip) 内的 `development/annotation/submission.py`，只用Python标准库，Python 3.10及以上可运行，不需要安装OpenCV。以下命令在仓库根目录执行，将 `<工具包目录>` 替换为解压得到的 `pingpong-annotation-tool` 绝对路径。使用现有标注虚拟环境时，把python替换为其python.exe路径；工作记录目录也应改为自己的实际位置。新窗口可直接将结果导出到仓库的 labels。

**现有工具初标导出：**

```powershell
python "<工具包目录>/development/annotation/submission.py" export --workspace outputs/team_annotation/A/03_007 --round initial --output-root labels
```

这是旧CLI导出方式，默认v1，生成labels/A/03_007/initial。CLI手动更新需递增--label-version。新窗口导出v2，自动判断内容变化、升版本，备份在工作目录_export_backups。

本地工作记录也可以使用CLI导出v2：

```powershell
python "<工具包目录>/development/annotation/submission.py" export --workspace outputs/local_annotation/A/03_007/initial --schema-version 2 --auto-version --output-root labels
```

内容不变不重写；不同输入不能覆盖。完整目录暂存、备份与回滚，失败保留上一版。草稿也可以导出，completed明确区分。

**现有工具复核导出：**

```powershell
python "<工具包目录>/development/annotation/submission.py" export --workspace outputs/team_annotation/B/03_007 --round review --review-of labels/A/03_007/initial --output-root labels
```

**自制工具生成复核来源字段：**

```powershell
python "<工具包目录>/development/annotation/submission.py" reference labels/A/03_007/initial
```

把输出对象写入自己metadata的review_of。

**提交前检查：**

```powershell
python tools/validate_labels.py validate labels
```

默认检查文件、类型、范围、重复记录、目录和复核来源；分配校验需显式提供 `--tasks data/manifests/tasks.json`。它不证明人工判断正确，也不表示所有任务完成。显示submissions和completed数量，便于区分部分提交和完成。

本机有这些提交对应的输入数据时，再加--verify-inputs及显式--tasks提供输入路径映射，核对哈希；输入在其他根目录则加--data-root。组长对最终用于分析的标签加--require-current-reviews，发现初标改动导致复核过期时退出失败，先重做相关复核。一般进度提交允许保留过期复核，并明确显示STALE_REVIEW警告。

标准文件包可在本地窗口“导入标准结果”：同一成员接续自己的结果，另一成员仅可用已完成初标创建独立复核。恢复自己的review还需绑定版本的完整初标包。导入核对输入哈希和实际解码信息，不覆盖已有工作记录。旧团队入口仍可使用session交接。

复核导出在目标树缺少来源初标时附带固定初标包，已有来源保留。组长汇总后用--require-current-reviews检查过期版本；工具不会替换新版初标来迁就旧复核。

## 9. GitHub提交流程

建议组长建立Private仓库并邀请三名组员。私有仓库协作者的邀请方式见[GitHub说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)。仓库地址与成员姓名由组长补充。

**上传范围**：labels下的人工交换结果、团队自行编写的代码和说明，以及 `tools/pingpong-annotation-local-20261007.zip` 标注工具发布包。视频、截图缓存、原始对象JSON、公开GT、官方权重、其他压缩包、解压后的本地标注工具、虚拟环境、原session和密钥保留在本地。根目录.gitignore已按此范围设置；不要强制添加被忽略材料。

人工补标也应遵守赛事对数据和衍生成果共享的约定。当前未查到官方材料的明确再分发许可；Private只是访问设置，不能代替赛事授权。上传者应有权提交相关内容，见[GitHub内容条款](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#d-user-generated-content)。本轮先按内部私有协作准备，公开发布前由组长核对赛事规定。

**用Git提交：**以下假设已克隆团队仓库并具有权限。

```powershell
git switch main
git pull --ff-only
git switch -c annotation/A-batch01
python tools/validate_labels.py validate labels
git add -- labels/A/03_007/initial
git diff --cached
git commit -m "annotation: A initial 03_007 v1"
git push -u origin annotation/A-batch01
```

分支名和路径换成自己的。查看暂存差异后确认只有本次标签；在GitHub建立Pull Request，写明成员、视频列表、完成/未完成、疑难和本次修改。B检查格式及样例，A合并。两人只修改自己的目录；复核人不覆盖初标。

**不熟悉Git时：**在GitHub的Add file → Upload files上传自己的完整视频/轮次目录，检查最终路径，选择新分支并提出PR，不直接覆盖main。保留JSON/JSONL原文件，避免只上传一个ZIP。官方操作见[上传文件说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。

## 10. B和A怎样验收

- 文件能解析，路径、成员、轮次、编号与任务清单一致；帧号和坐标合法。
- 哈希与统一输入一致，实际帧数和尺寸已核对。首段人工对照原图，抽查后续片段。
- completed与人工检查情况一致；无事件有说明；疑难有具体原因。
- 初标与复核独立保留，复核来源可追溯，过期复核不用于当前一致性结论。
- 分歧由B整理、相关成员讨论；决定后的汇总另存，不覆盖初标和复核。

比赛提交文件仍按官方video_id、frame_id、x、y输出。这里的人工标注文件用于开发和诊断，不能直接当作比赛预测提交。
