# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 graph-execution-foundation。
用户授权持续实施、私有环境、有限实验、分阶段提交推送；不使用子代理，参考仓库只读。
FP16实现8c8824f及独立CPU资格已完成；本次证据提交后推送并继续。总体未完成，下一阶段按 [ROADMAP](ROADMAP.md)。

## 已完成与当前增量

CPU FP64/FP32 的一般 SettleGraph、结构化表示、五记忆与正式发送/评分配置、
实际 packed/VJP/计算共享、standalone 训练及完整数值 checkpoint 的分阶段证据
从 [证据索引](evidence/README.md) 导航，不把各定向资格当作全矩阵。

[设备资格](evidence/core5-device-20261027.md)固定实现470ce06/tree6cf3e7a6cd9b2d5c00960b59b6132146751531fa：
CPU200项；NPU Python/native mixed-a/b各2、mixed-c各19组合，显式同步两卡各2组合通过。
native profile31531kernels/1641hostscalar/1372memcpy，Python开发profile28301/2733/1243，
均32candidate/31Full/44messages、20设备AiCPU；只回答归属调查，不是吞吐排名。

FP16未提交增量：同一seeded FP32参数量化、显式FP32评分；RMS/L2/scaled-RMS与KV Attention
内部FP32一阶运算；int64先相减再作衰减与位置计算，避免half长间隔溢出。
原始payload/state仍half，时间戳int64，支持/None/路由不放宽。
独立literal Jacobian、超过2^54的时间戳、五记忆完整记录/VJP、四步FP32 master与checkpoint
见 [精度契约](weighted-precision.md)。GDN显式SGD单列，默认AdamW没有自动替换。

最新开发任务 tide-core5-half-policy-dev-20261027 已inactive/success/exit0：
99 passed、2个具名strict xfailed（不计通过）、1条测试读取警告已改detach。
source470ce06 + frozen tree d4963248ac228886e47693db22d09bfed76eac72；
output artifacts/jobs/core5-half-policy-dev，status.json/development.json/task.log/build-manifest.json齐全。
此前half-normalization-dev为175passed/2failed、half-clocks-dev为95passed/2failed，均保留failed。

## 未解除的真实失败

- half GDN AdamW：Python/native同一输入的nodes.5.kernel.conv.weight更新严格差异约.001984。
  显式replay、保持标量舍入顺序的诊断仍失败；未放宽容差。具名strict xfail只接受该字段的
  GDNAdamWParityError，其他错误仍失败。诊断保存在artifacts/diagnostics/core5-half-gdn-*。
- 默认异步两卡Python完整VJP：CANN invalid event；core5-peer-event最小复现与current0/1/serial失败保留。
  显式ASCEND_LAUNCH_BLOCKING=1通过不解除默认异步限制。
- standalone SDK24图资格在fc/affine第三轮evidence失败，max_abs_error3.09944e-6。
  前两轮forward/grad/update通过；step0 evidence_weight差约2.91e-7、step1约5.59e-7，
  head.bias小梯度差经AdamW累积。mixed-a同样失败；未执行图不推定通过。
  诊断core5-sdk-affine/core5-sdk-affine-drift与相应jobs保留。SDK实际两图四步开发通过另列。

## 下一动作与任务状态

CPU固定资格 tide-core5-half-qualify-20261027 已inactive/success/exit0：
[99passed/2strict-xfailed](evidence/core5-half-20261027.md)，未计为完整FP16矩阵。
bootstrap干净8c8824f/tree691e16c9158100f702901d8b002ad3cd8d46cfc3，无reader；
output artifacts/jobs/core5-half-qualify，manifest与日志已独立保存；device/sdk仍470ce06。
1. 当前未提交设备入口草稿：显式dtype/optimizer/loss-scale、FP32 masters，恢复raw/master严格逐值；
   tests/test_weighted_device_precision.py待开发门禁。冻结新快照后执行该测试入口，接着实际NPU half。
2. 扩展check_weighted_device.py显式dtype/scoring/master/optimizer；实际NPU half训练恢复，
   不用CPU或旧core-3 half资格代替。继续SDK、异步peer失败定位。
3. standalone half masters/checkpoint、weighted resident、CUDA源码/构建/真机分账、有限选型性能与最终追溯仍待完成。

更新worktree前检查所有关联unit终态和status.json；活动任务只读冻结源码。
保存build manifest及源码/二进制摘要后才允许复用构建；新任务新输出，不覆盖失败。

## 环境与命令

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；zlong，约3.3TiB空闲。
现行规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与 /etc/modulefiles；旧指南路径不存在。
Python shell：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate。Python3.12.13/Torch2.12.0+cpu/Torch-NPU2.12.0.post2/CANN9.1/GCC10.3.1。
SDK单独shell：module load libtorch-npu/2.10.0-cann9.0.0；ACL_OP_INIT_MODE=0；
PATH前置$LIBTORCH_ROOT/../../../../bin提供匹配NumPy。未修改公共包。
ATen/BLAS单线程、2build workers、background.slice/Nice10、有限时长。
检查：python scripts/status.py；systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus；
读取output/status.json和task.log。停止：systemctl --user stop UNIT。
不恢复旧队列，不重跑历史全矩阵；失败保留，CUDA构建和真机证据分开。
