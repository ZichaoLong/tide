# TL-REQ-001 — FP32 稠密发送的独立参数更新超差

状态：下游已记录，尚未向基座交付/提交外部issue。本文件是唯一需求记录；基座与其他项目保持只读。

来源：实验实现`0d7ec5bdaa34556df9c93886e95f55fa5c9876ab`，固定基座`85d161d5c9148eac2b164f89a0ba4204b4331898`，数学`0770593838a37c790a568226930567ed60f99b83`；安装wheel SHA256见dependencies/foundation.lock.json。

目的：澄清或修复小型FC/HB、KV/qry、regional_total/Shannon发送在FP32端到端任务中，两步AdamW更新后的独立状态资格。输入/embedding/CE/head属于实验；候选用公共packed eager，独立锚用`tidegraph.reference.run`。没有复制执行核心。

复现：在本项目匹配私有环境和固定源码中运行：

```sh
python scripts/check_core5.py --device cpu --dtype float32 --send dense \
  --temperature .25 --adam-eps 1e-6 --output-dir NEW_OUTPUT
# NPU仅在站点队列分配后将device改为npu:0。
```

CPU dense原eps1e-8和新eps1e-6分别失败；NPU eps1e-6 step0 `regions.1.head.bias` 最大差5.334615707e-6，原atol1e-6/rtol1e-5失败；step1 descriptor evidence亦有有限超差。首步输出、loss、输入/参数梯度与optimizer槽通过。CPU FP64同型小检查通过。完整原记录与逐组件失败见docs/evidence/startup-checks.json。

未知：这是浮点运算顺序、近零梯度与Adam尺度的交互，还是实现路径偏差，尚未定位；不预设修复方式。另有原基座NPU综合根2.861022949e-6已知失败，不将二者自动合并为同一根因。

验收：原配置及严格阈值保持，固定候选安装对完整输出/私有状态/路由、输入及共享参数梯度、两步optimizer参数和槽、续接进行独立检查；如仅能给出有限域或不同数值策略，明确列出不支持条件和新的配置身份。应用精确恢复另验。

目前替代：A使用CPU FP64；B在明确标记的FP32有限数值探索下考察任务学习，不能宣称严格数值等价。该需求不阻塞已通过的独立CPU64任务或原骨干继续训练对照，不授权修改/分叉基座核心。
