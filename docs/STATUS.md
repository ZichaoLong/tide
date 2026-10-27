# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 graph-execution-foundation。
用户授权持续实施、私有环境、有限实验、分阶段提交推送；不使用子代理，参考仓库只读。
FP16实现8c8824f及独立CPU资格已完成；已推送6b7687b，继续设备入口。总体未完成，下一阶段按 [ROADMAP](ROADMAP.md)。

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

[01689c0设备half固定资格](evidence/core5-half-device-20261027.md)已全部终止通过：CPU12项，
Python/native各mixed-c AdamW4+SGD2、mixed-a/b各1、显式同步两卡各1。
native half profile35945kernels/1222hostscalar/1368memcpy，32candidate/31Full/44messages，
20设备AiCPU；非吞吐排名。全部命令/环境/输入及结果哈希已整理入证据JSON。
device保持干净01689c0/treefe3dc71f67b2a7520e3316a57b8cdf52c0f854c1，无reader；sdk仍470ce06。

standalone master/codec v2增量尚未提交（git status）：显式OptimizerPrecision、FP32 masters与scale，
Half原值、schema1读取/schema2写入、exact raw/slot恢复、独立Torch公式和畸形v1/不一致master拒绝。
CPU开发任务tide-core5-native-master-dev-20261027正在运行，不计通过；
bootstrap source01689c0+tree0f337e7cfa7d4fa79b0d468b5519d799cf6a8719，活动时不可编辑。
launcher .local/core5-native-master-dev.sh，output artifacts/jobs/core5-native-master-dev，
2workers/1800秒；test_cpp_master_optimizer/optimizer/checkpoint/weighted_training四入口。
主工作区另有晚于冻结的测试追加：有效checksum但master不一致的加载拒绝；standalone结果显式精度字段。

下一命令：systemctl --user show tide-core5-native-master-dev-20261027 -p ActiveState -p Result -p ExecMainStatus；
读取output/status.json、development.json及task.log。修复真实失败后冻结最终草稿并运行相关门禁，
再提交实现、固定资格与单独证据。不能将01689c0设备通过延伸到新的native optimizer/格式。
随后继续standalone NPU、weighted resident、CUDA源码/构建/真机分账、有限性能与最终追溯；总体未完成。

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
