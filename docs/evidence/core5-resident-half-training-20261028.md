# Core-5 FP16 公共全图与训练资格

固定实现 `21ed86d019d5dcbe3bffa363be02506433e23a7f`，数学上游
`0770593838a37c790a568226930567ed60f99b83`。2026-10-28审阅。
机器可读身份、binary/loader/日志/XML hash见[JSON](core5-resident-half-training-20261028.json)。

| 独立门禁 | 已核验结果 |
| --- | --- |
| CPU开发相关门禁 | 32通过、0 skip；testcase合计3.452秒 |
| standalone NPU，FP32/FP16各自 | 完整推理21图、完整配置15项、公共训练5配置20更新，每轨迹3个连续窗口 |
| Python-owned NPU | 52通过、0 skip，pytest165.29秒；训练18、图执行28、独立KV手算根6 |
| CUDA standalone | 265步选定构建、loader及三入口help通过；实际选定67条CUDA编译、全部配置161条命令均为sm80/90/100 |

三个持久任务均为passed/exit0；对应unit复核inactive/success/0，冻结工作区仍干净。
NPU standalone为SDK Torch2.10/CANN9.0，物理11→npu:0；Python-owned为Torch2.12/CANN9.1，物理9→npu:0。
各自构建与core身份独立校验，没有混用二进制。每任务1800秒、2个build worker、ATen/BLAS1、
Nice10、独立background.slice，设备排队180秒、执行600秒。CPU codec与安装消费者没有在本批重跑。

后续检查器审计补注（2026-10-28）：standalone训练的最后master比较曾引用
`master_registry.owners()[j].value`中临时vector的成员，生命周期无效。本批实际退出状态不变，
但该一处master数值断言不能作为可靠独立证据；Python master检查与其他完整记录/VJP检查不受这处影响。
当前修复持有owners vector，并重做standalone训练/peer定向资格，状态见[STATUS](../STATUS.md)。

精度采用[公共训练契约](../resident-weighted-training.md)：raw与实际发送payload按FP32或FP16存储，
局部算术、Describe/control、舍入前Full G、所有伴随和optimizer master/槽使用FP32；
时间、计数与原始缓存坐标为int64。存储Cast声明identity VJP，参数发布先half舍入再拓宽计算bank。
checkpoint同时保留实际named值和master，恢复验证二者一致。half cotangent拒绝。

完整记录包括所有状态字段、候选/活动支持、控制、逐物理端口、消息/fiber/pending与元数据。
FP16完整记录容差为rtol=.02/atol=.001；梯度仍为3e-5/3e-6，named参数更新仍为1e-5/1e-6。
离散、物理身份、int64和None/连接零保持严格。恢复核对全部逻辑raw字段，包括过期但仍保留的条目；
未使用journal行和容量padding没有数学值，不要求其未初始化内容相同。

独立参考使用原CPU数学函数和独立时间递归/Greedy调度，仅在声明的raw与发送边界存储舍入。
生产代码和解码不调用该参考。公开Full根另由独立公式计算舍入前G，不能用发送payload代替。
单节点KV的key/summary/weight根、RMS Jacobian、参数外积与输入伴随另有手算锚点。

standalone包含FC、同秩HB、非对齐DAG、反馈PDG、异构记忆、重复端口、空窗、完整字段和左切面根、
SGD/AdamW、多步/冻结参数、None/零、梯度累计、切面恢复及生命周期拒绝。
Python八种family/topology/optimizer轨迹各两dtype、每轨迹三更新；包括磁盘切面和KV/FC新进程恢复。
Python警告为共享只读安装的属主提示、上游弃用/转义/fork提示；真实计算与恢复检查全部完成。

旧失败保持：`core5-half-flow-{npu,cuda}-qualify`因已存在build目录被构建入口拒绝，未编译；
`core5-half-flow-npu-qualify2`在31e017f的FP32全图后，FP16旧eager参考触发
`weighted allocation requires FP32/FP64 scores`，配置门禁未运行。这三项仍为failed/exit1，
原status/log hash在JSON中，不能用本批通过改写。新参考匹配已有FP32算术精度契约，并非放宽梯度或路由检查。

复现入口与三个测试文件见[公共训练](../resident-weighted-training.md)。本批standalone命令只选
`--checks weighted-flow weighted-flow-config weighted-training`，Python选
`test_resident_weighted_training.py test_resident_weighted_flow.py test_resident_weighted_storage_roots.py`。
构建和运行必须来自新的固定checkout、独立目录与匹配工具链。

本报告不认证跨卡、性能或新安装消费者。所有CUDA真机范围仍待硬件；原eager FP16 GDN AdamW、
默认异步eager两卡及SDK2.10的严格见证保持独立限制，本报告的常驻精度路径不解除它们。
