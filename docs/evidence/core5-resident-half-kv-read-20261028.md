# 常驻 KV 半精度 raw 读取资格

固定实现 `db85db6e2e91559b71915200298203e6f313982c`，上游 `0770593`。
[机器记录](core5-resident-half-kv-read-20261028.json)保存源码、binary/loader、运行及队列摘要。
契约见 [sum](../resident-weighted-kv-sum.md) 与 [qry](../resident-weighted-kv-query.md)。

CPU相关27项通过、无skip。Standalone NPU 的sum/qry各FP32/FP16全部通过；
每配置两种窗口、120重放行、四个独立Mem根，覆盖原始字段、输入、共享参数和保存版本VJP。
FP16只描述实际raw存储；窗口、算术、qry输入/参数、四个返回及伴随为FP32。
独立逐条CPU公式直接消费量化后的raw，不调用候选窗口、pool或query程序。

另以精确断言核验最小half次正规key/value/weight拓宽，正的短期FP32证据低于half下限
仍保留，以及四根连接和大int64年龄。FP16 cotangent拒绝，None/零检查未放宽。
物理NPU 9→npu:0，LibTorch-NPU2.10/CANN9.0；持久passed/exit0、
systemd inactive/success/0和干净源码、二进制摘要均复核。

CUDA同组件build/loader/help通过；所选依赖23条CUDA编译命令，全部160条已配置命令
均含80/90/100。没有GPU真机、Python-owned FP16、完整图训练或性能结论。
复现命令在机器记录中；不要求找回本次原始运行目录。
