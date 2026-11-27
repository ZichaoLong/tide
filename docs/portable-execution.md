# 在本机和 ModelArts 接续冻结的 R3 任务

项目负责冻结原始任务、源码、数据、科学门禁和检查点。账户的 `maflow batch` 负责批次、地点、作业、尝试编号、区域制品和资源释放。两者使用同一个任务 ID 及原始 `task.json` 字节的 SHA-256。

## 准备已有任务

在维护分支使用已安装 NumPy 的项目解释器运行以下命令。把两个修订替换为任务采用的实现。冻结的数据目录包括 R3 视图及其原始 article corpus。阶段 B 还会复制任务引用的 gate、manifest、report 和确认授权。另有要保留的外部证据时重复传入 `--evidence 文件`。

```bash
.venv-pilot-r3/bin/python -B scripts/make_tide_batch.py \
  --output "$BATCH_DIR" --project "$PROJECT" --batch "$BATCH" \
  --foundation ../graph-execution-foundation --runq ~/tools/runq \
  --experiment-revision "$EXPERIMENT_COMMIT" --foundation-revision "$FOUNDATION_COMMIT" \
  --task "$TASK_JSON" --data "$R3_DATA" --site modelarts --backend npu \
  --nodes 1 --ram-mib 8192 --hbm-mib 8192 --worker-seconds 420 --run-seconds 900
```

按每项实际配方填写申请量与时限。对同一冻结来源的多项任务可重复使用 `--task`。本机使用 `--site local`，`--python-command` 可指定目标解释器。混合运行时，把分别准备好的组放在同一个计划中，为组指定各自的地点、输入目录和唯一任务 ID。云组内 runq 使用已获配的实例与卡，本机组沿用账户 runq 已分配的设备。运行、查询、接续从 `maflow batch --help` 接手。

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

完成项的 `evidence` 可交给 `pilotlab.gate3.read_evidence` 及上层科学审阅。复核时用本地副本核对原 worker 结果声明的检查点 SHA-256。原 worker 的绝对路径与原始结果文件仍可从制品追溯。

## 两份短作业验收

生成时省略 `--task` 和 `--data` 会建立一份独立的 R3 迁移验收批，使用固定的合成整数 token 和两个真实项目模型。第一份双实例各跑一个任务。TIDE 在第一个完整更新保存后中断，独立 Transformer 完成三个更新。第二份单实例仅恢复未完的 TIDE 任务，累计完成三个更新。每项任务的原始 target 均为 12 个 token。首份可留最多 300 秒检查两个节点，向本作业 OBS 写入 `control/client-confirmed.json`，内容为 `{"attempt":"当前作业名"}`，可提前结束这个窗口。两份作业分别使用自己的截止。验收只覆盖这组小输入的执行、转移和接续。

云端 `products/tide-queue.json` 记录队列终态，各 `products/tide-node-<序号>.json` 记录节点进程的 PID、Linux 启动 ticks、队列实例身份和排空时间。任务回执记录所在物理板、芯片、独立 worker 的 PID 及其启动 ticks。
