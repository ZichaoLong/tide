# Core-5 Read/Full 批量 VJP CPU 资格 — 2026-10-27

固定实现/测试 `67854b306a97a1e39d6c4ae995906963507bdc7b`，上游 `0770593`。
**647 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、工作退出 0 与服务终态 inactive/success 已核对；[命令/环境/哈希](core5-query-vjp-20261027.json)。
[数学对应与公开能力](../weighted-query-vjp.md)说明一阶边界及字段解码。

Python 和独立 C++ 保留实际 Read/Full 批量图，反向不重放 scalar Read/Mem/Full。
参数树保留别名；真实浮点/int64 状态、content 和 control 构成私有图输入。
按结构、Read⁰/实际快照、KV 正权重分支及已定义的字段 cotangent 模式隔离行根，
保留 None/连接零、重复输入、共享参数贡献和原始版本检查。Read 自动采用已声明能力；
Full 通过 `full_autograd="batched"` 选择，默认 replay 保留。
Read 的 Mem 实际合批，有限描述头保留逐行数值次序；不宣称全部 Read 算子已融合。

局部检查覆盖五种记忆、适用读出、完整字段/槽/行独立根和方向差分。
HB/反馈 PDG 两样本验证三步 AdamW、含 pending 的 checkpoint、换 owner 恢复及保留图切点。
standalone C++ 通过 literal SiLU 和单项带权 KV 导数手算；该组件证据不替代独立 C++ 完整训练矩阵。
相关 Read/Full 契约、Aggregate/State、FC/HB/非对齐 DAG、正式评分/发送与循环回归也在本次范围。

开发编译失败 tree 5e93638676346279ab69d7b4182b7b8aeb2f991c：helper bind 被 ADL 解析到 std::bind；
改名 bind_queries 后构建通过。开发 tree 35145d0b86090b2c77f472367b6fd8e9443bc6df 为 **646/1**；
FP32 native gated_delta/sum 混合字段根 input.3 差 1.43e-5。
Python MatchScore residual 投影改为教材/native 的列向量次序后相关 137 项短测通过。
数学函数、输入、dtype 和容差未改；两次失败的源码/日志/终态独立保留。

复现：检出固定实现提交，按 [development](../development.md) 构建，再执行 JSON 中的 pytest 测试列表；
使用 `--dtype both` 和新的 basetemp，`TIDE_BUILD_DIR` 指向当前构建。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1，单线程 ATen/BLAS、两个 build workers。
服务 tide-core5-query-vjp-qualify-20261027 在 background.slice，Nice=10，15 分钟上限；
原始记录 artifacts/jobs/core5-query-vjp-qualify，build manifest 已独立保存，复现无需旧目录。
本报告不是区域批量、设备真机/常驻、性能或全仓库矩阵验收。
