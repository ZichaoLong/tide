# 在本机和 ModelArts 接续冻结的 R3 任务

项目负责冻结原始任务、源码、数据、科学门禁和检查点。账户工具 `maflow` 记录执行地点、作业、尝试编号、区域制品和资源回收情况。两者使用同一个任务 ID 及原始 `task.json` 字节的 SHA-256。

## 贵阳 R3 正式续跑

这次迁移使用两个单节点 ModelArts 作业。每个作业申请 8 张物理板、16 颗逻辑芯片，各运行自己的 runq。维护分支中的 TIDE 控制器保管全批任务、尝试和科研结论。登记身份为 `tide-learning-lab-c1dd49b83311`。11 月 28 日提交时 maflow 在平台清单中写入了路径别名 `tide-learning-lab-eb9313f37e43`，两份原始清单与统一项目 ID 的对应关系已登记。两份作业的输入包按 SHA-256 冻结。[作业 ID、输入摘要与预算](evidence/pilot-r3-guiyang-production-20261128.json)

在本机从项目状态入口查看上次平台采样时间、Stage A 选方数、CPU/NPU 首次恢复和每个节点的队列计数。可按作业 ID 向贵阳重新采样。进入排队状态的作业在平台分配整节点后才开始计卡时。

```bash
python scripts/status.py
maflow job show guiyang/c4a30523-cc7d-42c6-9b60-0ef3f4143ccc --json
maflow job show guiyang/f0f47300-b53c-441e-95b4-91e245f0b5d0 --json
maflow job queue status guiyang/tide-r3-20261128-1 --json
maflow job queue status guiyang/tide-r3-20261128-2 --json
```

恢复先用 2237 的 NPU 第 1,312 步和 2907 的 CPU 第 1,770 步。每项都要验证旧末态、执行下一步有限更新、保存后用新的模型和优化器对象重载。两份回执到齐，控制器就会持续向已经获配的节点分派后继工作。它保留原任务的种子、后端、配方、累计曝光和逐主体评估预约。每张物理板每小时计一张卡时，逻辑芯片、CPU 工时和同卡并发数另列。长时或大 HBM 任务可在 runq 保护一个芯片开工窗口，其余任务根据真实余量和预计结束时间补位。[运行时迁移](../pilotlab/runtime_migration3.py) · [派发与预约](../scripts/tide_cloud_dispatch.py)

客户端的用户服务为 `tide-r3-guiyang-controller-20261128.service`。控制器按任务 SHA 和尝试号在华东预约一次。验证到节点的 SSH、挂载及 runq 身份后，它把一批原始任务送入该节点的私有 SFS 队列；只有 OBS 可达时，它使用作业内置的 OBS 派发入口。每次尝试记录所用入口，断线后用同一任务键重试。若队列已结束、项目结果迟到，控制器可从该节点的原始产物补回回执。维护者可在同一分支提交控制器修订后运行 `bash scripts/tide_cloud_service.sh`，在途 worker 由所在的贵阳作业继续管理。[服务入口](../scripts/tide_cloud_service.sh) · [队列派发](../scripts/tide_cloud_submit.py) · [回执恢复](../scripts/tide_cloud_relay.py)

华东权威前缀为 `east:/tide-learning-lab/production/r3-20261128/`。`control/rollout.json` 绑定两份平台作业，`control/status.json` 给出最近观察，`control/state.json` 记录已预约与已收的尝试。`dispatch/` 保存原任务字节和有序的断点历史，`attempts/` 保存原始结果、运行时计划和逐项产物凭证。热输入经 OMS 进入贵阳，每个节点按摘要取用。新检查点在贵阳读取正文复核，随后由 OMS 回传华东。`science/current.json` 标识最近一次原生选方、评估报告和门禁快照。新客户端可以先取以下轻量文件。

```bash
maflow fs cp east:/tide-learning-lab/production/r3-20261128/control/rollout.json ./rollout.json --verify sha256
maflow fs cp east:/tide-learning-lab/production/r3-20261128/control/status.json ./tide-status.json --verify sha256
maflow fs cp east:/tide-learning-lab/production/r3-20261128/science/current.json ./tide-science.json --verify sha256
```

Stage A 包含原 270 个主体、74 个早期主体及匹配对照。旧源沿用的 4 个配方保留原证据身份。控制器从冻结的 A1/A2/A3 轨迹逐步生成下一批。A 门禁还要核对来源、恢复链、初始化、匹配质量、活动量、评估预算和可用卡时。满足后进入 B 正式训练，再按原规则选 6 个主体及对应对照做多种子复跑，最后对列入清单的正式终态做独立确认评估。B 的 NPU 预约引用同来源、同形状的实测峰值，缺测的形状在有容量的芯片上先做有限的准备测量。准备任务沿用原有的 48 物理卡时和 64 Mi token 上限，其已用量跨迁移累计。[阶段推进](../pilotlab/cloud_campaign3.py) · [独立门禁](../pilotlab/gate3.py)

## 准备已有任务

在维护分支使用已安装 NumPy 的项目解释器运行以下命令。把两个修订替换为任务采用的实现。冻结的数据目录包括 R3 视图及其原始 article corpus。阶段 B 还会复制任务引用的 gate、manifest、report 和确认授权。另有要保留的外部证据时重复传入 `--evidence 文件`。

```bash
.venv-pilot-r3/bin/python -B scripts/make_tide_batch.py \
  --output "$BATCH_DIR" --project "$PROJECT" --batch "$BATCH" \
  --foundation ../graph-execution-foundation \
  --experiment-revision "$EXPERIMENT_COMMIT" --foundation-revision "$FOUNDATION_COMMIT" \
  --task "$TASK_JSON" --data "$R3_DATA" --site modelarts --backend npu \
  --nodes 1 --ram-mib 8192 --hbm-mib 8192 --worker-seconds 420 --run-seconds 900
```

按每项实际配方填写申请量与时限。对同一冻结来源的多项任务可重复使用 `--task`。本机使用 `--site local`，`--python-command` 可指定目标解释器。混合运行时，把分别准备好的组放在同一个计划中，为组指定各自的地点、输入目录和唯一任务 ID。云组内 runq 使用已获配的实例与卡，本机组沿用账户 runq 已分配的设备。

生成器调用账户的 `maflow admin runq freeze --directory PAYLOAD`，将当前 runq 发行包、Git commit 及 SHA-256 装入 `payload/.maflow/runq/`；指定候选版本时传生成器 `--runq-manifest /路径/release.json`。云端启动器优先解包该冻结版本，TIDE 通过 `MAFLOW_RUNQ` 执行并核对身份。只有项目驱动向自己的队列提交、排空并封存科学回执；`maflow job queue status` 可查看该队列。需要由客户机持续追加独立命令时，按账户工具的约定创建可追加队列会话。以前已封存的批次继续使用各自的快照。

以下命令中的 `SITE` 取本批约定的 `modelarts`、`local` 或 `hybrid`。云组可在生成计划时用 `--target` 指定计算目标，`--spec` 指定资源规格，`--authority` 指定存放整批计划及结果索引的 OBS 别名。多目标计划中，每个云组分别固定目标与资源申请。`project run --dry-run` 显示依据当前回执预计新增的分配，去掉该选项后提交。新客户端用 `project list` 发现各地项目及批次，用 `项目/批次` 定位结果。同一引用在多处登记时以 `--authority` 指定权威存储。

```bash
maflow project run --site "$SITE" --plan "$BATCH_DIR/plan.json" --dry-run
maflow project run --site "$SITE" --plan "$BATCH_DIR/plan.json"
maflow project list --site "$SITE" --project "$PROJECT"
maflow project show "$PROJECT/$BATCH" --site "$SITE" --json
maflow project collect "$PROJECT/$BATCH" --site "$SITE"
maflow project resume "$PROJECT/$BATCH" --site "$SITE" --dry-run
maflow project resume "$PROJECT/$BATCH" --site "$SITE"
```

`payload/frozen/freeze.json` 将原路径映射到包内相对路径，并列出文件大小和 SHA-256。原始 JSON 中的路径、原始字节及源码身份继续参与校验。运行适配只在该 worker 子进程内将已登记的路径解析到本地副本。原任务声明的 Python、依赖、后端与训练轨迹约束均由冻结的 worker 再次检查。原配方绑定的运行时版本与云镜像不兼容时，在项目中单独审阅新配方。已有任务继续以冻结字节为准，`release.json` 还登记适配程序、runq 与用于交接的 Git 修订。

## 恢复与项目复核

活动检查点先在本次作业的 SFS 发布，随后复制到本次作业 OBS。所有制品读回核对后，才发布 `products/tasks/<任务 ID>.json`。独立尝试使用各自的工作目录和输出前缀。接续时，账户层将上一尝试的 checkpoint 复制到新作业输入，项目按 SHA-256 和长度复核后换名装入新工作目录。冻结 worker 还会验证底层状态摘要、模型、优化器、私有状态、数据流和 RNG。恢复回执记录所选代、起始步以及实际前进步数。设备序号可以变化，同一轨迹的后端必须符合原契约。

项目 worker 和队列各自写执行状态。计划中断会使训练 worker 在完整更新处收到 SIGTERM，留下完整检查点及 R3 原始的 `budget_exhausted/signal`，项目回执标为 `interrupted`。已完成项还须通过项目原有的健康、暴露量、验证损失和恢复检查。跨作业精确恢复本身只验证了加载与后继执行，数值轨迹比较沿用项目独立的实验范围。

将各次作业的 `products/` 按原相对布局下载到独立目录，重复传入各次目录即可重算项目汇总。相同回执重传会去重。汇总会逐文件验证哈希和长度，报告缺项、被改写的证据、同一尝试的不同回执、多个完成项和缺失的检查点前驱。

```bash
python3 -B scripts/tide_batch.py aggregate --frozen "$BATCH_DIR/payload/frozen" \
  --attempt-root "$DOWNLOADED_FIRST_JOB" --attempt-root "$DOWNLOADED_RESUME_JOB" \
  --output "$REVIEW_DIR/aggregate.json"
```

换到新客户端时，在已安装账户 `maflow` 的机器进入实验仓维护分支。下面的命令从所选权威存储取回默认 `tide-workers` 组的全部尝试和冻结输入，适用于单地点及混合批次。独立 CPU 检查、其他项目模型等同批分组各自使用其项目的验收程序。

```bash
mkdir -p "$HOME/.local/state/tide-reviews"
REVIEW_DIR=$(mktemp -d "$HOME/.local/state/tide-reviews/$BATCH.XXXXXXXX")
maflow project collect "$PROJECT/$BATCH" --site "$SITE" --group tide-workers --to "$REVIEW_DIR/download" --inputs --json > "$REVIEW_DIR/checkout.json"
python3 - "$REVIEW_DIR" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

review = Path(sys.argv[1])
checkout = json.loads((review / "checkout.json").read_text())["checkout"]
source = checkout["source_archives"]["tide-workers"]
subprocess.run(["tar", "-xzf", source, "-C", str(review), "frozen"], check=True)
command = [sys.executable, "-B", "scripts/tide_batch.py", "aggregate", "--frozen", str(review / "frozen")]
for directory in checkout["attempt_roots"]:
    command.extend(["--attempt-root", directory])
subprocess.run(command + ["--output", str(review / "aggregate.json")], check=True)
PY
```

`$REVIEW_DIR/aggregate.json` 记录任务状态、缺项、冲突以及已验证的本地证据路径。保留该目录即可再次审阅。只关心特定任务时，可给 `collect` 重复传 `--task 任务ID` 以下载其完整尝试链。整组汇总会将未下载的任务列入 `missing`。

完成项的 `evidence` 可交给 `pilotlab.gate3.read_evidence` 及上层科学审阅。复核时用本地副本核对原 worker 结果声明的检查点 SHA-256。原 worker 的绝对路径与原始结果文件仍可从制品追溯。

同一 A2/A3 任务跨尝试时，`portable_attempts` 按顺序绑定每份实际结果、尝试回执和检查点。审阅读回每段的中断原因、私有状态摘要、真实起始步、token 暴露量和父尝试。阶段门禁据此恢复该梯级的最初起点，末次尝试的实际计数按原记录保留。

## 两份短作业验收

生成时省略 `--task` 和 `--data` 会建立一份独立的 R3 迁移验收批，使用固定的合成整数 token 和两个真实项目模型。第一份双实例各跑一个任务。TIDE 在第一个完整更新保存后中断，独立 Transformer 完成三个更新。第二份单实例仅恢复未完的 TIDE 任务，累计完成三个更新。每项任务的原始 target 均为 12 个 token。首份可留最多 300 秒检查两个节点，向本作业 OBS 写入 `control/client-confirmed.json`，内容为 `{"attempt":"当前作业名"}`，可提前结束这个窗口。两份作业分别使用自己的截止。验收只覆盖这组小输入的执行、转移和接续。

云端 `products/tide-queue.json` 记录队列终态，各 `products/tide-node-<序号>.json` 记录节点进程的 PID、Linux 启动 ticks、队列实例身份和排空时间。任务回执记录所在物理板、芯片、独立 worker 的 PID 及其启动 ticks。
