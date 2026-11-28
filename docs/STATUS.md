# R3 已提交贵阳双节点，Stage A 已选 268/344

北京时间 2026-11-28 20:09，两份贵阳八卡作业均为 `Pending`，平台子状态分别为 `Queuing`、`OSQueuing`。20:08 的父资源池采样含 16 个节点、256 颗逻辑芯片，各节点可分配芯片均为零。云端在训 worker 为零，本机旧队列在训和持卡数也均为零。账户控制器已在本机 `background.slice` 运行，先获配的节点可以独立开始。现场状态与采样时间用 `python scripts/status.py` 查看。[作业与队列入口](portable-execution.md#贵阳-r3-正式续跑) · [平台受理与发布凭据](evidence/pilot-r3-guiyang-production-20261128.json) · [资源池及两节点输入复核](evidence/pilot-r3-guiyang-dispatch-ready-20261128.json)

11 月 27 日本机释放算力时已选 262 个主体。本次原生重放纳入停派期间自然完成的结果，又选出 6 个，已选 268 个，剩 76 个。192 个原待派任务、69 个断点任务和 20 个新生成任务合计 281 个，其中 155 个 NPU、126 个 CPU。禁止读取本机任何 `.pt` 的全量演练已验证这 281 个请求、现有选方和 255 个仍可能被后继使用的历史训练端点。后续任务数随 A1/A2 晋级、A3 和匹配对照结果变化。74 个早期主体已有逐项裁定，4 个有限沿用，70 个在目标源码下按裁定补跑。[选方与父链](evidence/pilot-r3-guiyang-archive-readback-20261128.json) · [全量续跑演练](evidence/pilot-r3-guiyang-candidate-recovery-20261128.json) · [逐主体裁定](pilot-r3-deferred-dispositions.md)

贵阳私有镜像运行 Python 3.12.3、PyTorch 2.10.0、torch-npu 2.10.0.post4。前两条真实跨版本接续是 NPU 2237 和 CPU 2907。门禁要求两项各自完成一次更新及新检查点精确重载；当次核验通过 0/2。各 worker 保留原任务的源码、指纹、初评、种子和累计预算，另存实际运行时。两节点分别计 8 张物理板，未使用的原预算为 5,848.77 物理卡时，绝对截止为北京时间 2026-12-20 00:48:33。[原始发布、身份与预算](evidence/pilot-r3-guiyang-production-20261128.json)

华东 OBS 保存训练数据、断点和原始证据。14 个 OMS 作业已向贵阳交付全部 210 个热对象，共 19,237,056,320 字节。两份云作业已各自固定首步所需的 18 个对象，逐任务取用前还会校验内容 SHA。本机按物理块累计释放 491.79 GiB，20:03 的配额为 68.45 GiB，项目实占 24.47 GiB，其中 17.66 GiB 是等待真实跨版本恢复的热检查点。三份 ModelArts 测试批和独立资格测试共回收 101 个 OBS 对象、28,695,799 字节，对应 SFS 测试目录有释放回执。[本机配额与隔离恢复](evidence/pilot-r3-modelarts-local-storage-20261128.json) · [检查点恢复](README.md#冷检查点归档与恢复) · [测试清理回执](evidence/pilot-r3-guiyang-production-20261128.json)

旧私有 SFS 的 1,818 个压缩检查点已在华东逐个解压、核对原 SHA。20:09 已释放其 146.19 GiB 压缩数据并清理本机 801 条旧链接。恢复总索引为 `east:/tide-learning-lab/production/r3-20261128/recovery/index.json`，SHA-256 为 `b65c287a1beb58574ae58e646658eccf002d447d8f3a64746abda8fd8330a808`。可按原路径查找 149,041 条路径记录。[按原路径恢复](../scripts/restore_modelarts_archive.py)
