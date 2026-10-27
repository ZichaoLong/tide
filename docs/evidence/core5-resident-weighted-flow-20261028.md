# 完整 weighted 常驻推理资格

实现提交 `5efce32471f3b407b25fa669706e56e88436a3e9`；固定上游 `0770593`。
[公共接口、函数对应与复现](../resident-weighted-flow.md)；[机器记录](core5-resident-weighted-flow-20261028.json)。

单设备 FP32 推理：五记忆、三 Read 快照、十种发送及适用评分/key/来源绑定，
覆盖 FC/HB、反馈 PDG、非对齐 TimedDAG、一般 Settle 端口与 streaming/prefill。
独立 CPU Greedy 核对完整记录、控制、逐物理边/端口、raw state/history/pending，
并验证两段新 owner 和磁盘 checkpoint。解码仅恢复候选真实计算结果。
局部公式独立锚点沿模块契约追溯；共同调度对照不代替公式证据。

| 路径 | 固定提交实际结果 |
| --- | --- |
| CPU Python/native | 50项 / 22.91秒；C++ kernel FP64/FP32、两独立 DSO 配置检查通过 |
| standalone LibTorch-NPU 2.10 / CANN9.0 | metadata 5项；services、21图案例及边界、15配置、DSO检查通过；物理3→npu:0 |
| 独立安装消费者 | 公开头与 CMake 包构建/loader；NPU手算 `(c,G)=(m,z)` 与两段新 owner 通过 |
| Python-owned Torch-NPU 2.12 / CANN9.1 | 公共图18项 + metadata5项 / 24.14秒；物理1→npu:0 |

三个资格任务均 clean、inactive/success/exit0。复用构建逐源码/二进制摘要验证，
安装规则重配置不冒充重编译；原 manifest 与失败记录保留。
NPU数值阶段采用设备通知握手，窗口内仍由设备决定就绪、选择与推进。
独立 services 检查覆盖多轮、空窗口和重用，共享 workspace 为16,777,472字节。

开发中的 CANN task buffer full、旧增量对象、缺字段 fixture、跨 DSO 类型与末尾
缺失 `rg` 均保留原失败终态；后续通过不覆盖失败。CUDA旧开发因隐式 sm50 编译失败，
架构约束修复另行构建验收，本报告不声称 CUDA 构建或真机通过。

本资格不认证完整图 retained VJP、optimizer、训练恢复、FP16、peer 或吞吐。
