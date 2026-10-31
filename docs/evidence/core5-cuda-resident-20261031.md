# RTX3090 单卡 resident 与 eager 补验 — 2026-10-31

[固定来源、完整库存、逐例终态与命令](core5-cuda-resident-20261031.json)。实现固定
`32edb0f2b035ef2b732b6f27cf577f9e98058eb6`，数学固定`0770593838a37c790a568226930567ed60f99b83`。
环境同[首轮 eager 报告](core5-cuda-eager-20261031.md)：x86_64、RTX3090/SM86、Torch2.12.1+cu130、Toolkit13.0.88。
下表作业全部终止，无存活工作后代；测试耗时不是性能。设备时间线及有限性能另验。

## 构建与改动

- 新 core、Python-owned resident、standalone resident、安装头/库消费的 standalone benchmark 四阶段构建通过；56线程，显式SM86。
  cuobjdump核对control、weighted-flow、weighted-training产物只含sm_86 cubin。不同客户端实际执行仍分别计数。
- 6cacad0按字段首次出现顺序打包，TensorImpl只识别别名；独立逆地址反例和相关CPU放置110项通过，C++局部核FP32/FP64两个进程通过。
  该修复未解除双卡GDN精确恢复失败。
- 8a34f66用CUDA实际retained operands/metadata检查纯控制所有权，保留CANN workspace断言；循环组件预算扣除既有error flag后测试精确增量与one-byte不足拒绝。
- 32edb0f将已退役LH vendor JSON隐含依赖改为显式外部头；本次私有nlohmann_json3.11.2、44头文件，树hash
  `bd976a412fde243decb465a94d498bcc254bb5ad79be63a8f201ec9ba298ff34`。standalone runtime不链接Python，TF32策略显式关闭并记录。

## 单卡 resident 正式结果

`verify_core5_resident_single.py`通过：39个组件入口的56个dtype单元，加公共Python53项，零跳过。
库存由[core5_resident_scope.py](../../scripts/core5_resident_scope.py)定义；不是整个旧模块集合或三卡门禁的缩减通过。

| 层 | 实际检查 |
| --- | --- |
| 控制/队列 | int64、条件WHILE/SWITCH、闭包/ready、稳定选择、事务/广播、容量拒绝及完整状态 |
| 局部数学 | 五记忆、短卷积、投影、rotary、分配/十种发送/head/region、Aggregate/source、独立字段/参数VJP |
| 完整图/训练 | FC/HB/非对齐DAG/反馈等声明fixture，forward、stage/window/retained VJP、SGD/AdamW、FP32/FP16 master发布、跨窗及checkpoint |
| Python公共调用 | flow29、literal storage roots6、training18（含入口拒绝检查）；完整字段、参数/输入梯度、None/零连接、磁盘新owner；KV/FC另有新进程续接 |

FP16 resident按既有“half存储、FP32算术/伴随/master”契约核对独立CPU精度参考，不与eager half算术混为一谈。
实际本次独立调度与根检查见[语义映射](../semantic-map.md)及上述固定源码库存。
完整多卡门禁仍要求至少三卡及双向P2P/原生原子；当前容器不满足，未跳过这些条件。

旧`c368b45`库存56中54通过、公共53通过，整体仍失败；失败为旧control workspace和recurrent增量预算断言。
新来源通过只验收修复，不将旧失败改写为通过。

## Eager 补验与原严格失败

均保留原输入、AdamW、epsilon、loss scale16、数值阈值与同栈逐值恢复断言。
FP32禁用TF32；ATen/BLAS/interop/native worker均1。

| 范围 | 结果 |
| --- | --- |
| standalone mixed-c，新增独立输出/descriptor/control/state及连接零根 | FP32 23/24；FP16 22/24，两个总gate均失败 |
| scalar FP32，六代表×三客户端 | 18/18通过 |
| 普通双卡复制 FP32，六代表 | standalone6/6；Python native5/6 |
| 修改后native FP16六代表 | 6/6通过 |

代表覆盖五记忆及异构，实际用例名/拓扑分别保留，不把不同客户端fixture互相代替。
ordinary-copy成功不证明直接P2P、NCCL或图捕获，既有AccumulateGrad流警告仍保留。
纯PyTorch原19项及native原19项完整单卡范围见首轮报告；本次没有把未重跑范围计入新来源通过数。

- standalone FP32 `fc/ssd`仍在第三次AdamW后`regions.0.head.bias`失败：CPU
  `0.11622299998998642`、GPU`0.11622648686170578`，差`3.4868717e-6`，超原阈值。
- standalone FP16 `feedback/affine`第三次更新后`nodes.3.extra.kernel.read_out`第4个flat坐标：
  期望`0.0185699462890625`、packed实际`0.0199737548828125`，差`0.00140380859375`。
  CPU原fixture复现相同值；显式scalar在CPU/CUDA也同样失败，实际为`0.0204010009765625`。
- standalone FP16 `feedback/ssd`第三次更新后`nodes.1.extra.read_program.evidence_weight`第12个flat坐标：
  期望`0.01092529296875`、packed实际`0.01220703125`，差`0.00128173828125`。
  CPU原fixture复现相同值；显式scalar在CPU/CUDA各通过完整四更新、独立根和恢复，**packed原失败未修复**。
- native双卡`gated-hb`仍在第四步精确恢复的`nodes.0.kernel.read_out`梯度3/4坐标不等，最大`4.44e-16`。

以上FP16隔离来自同一新固定源码实际运行，不能归因为CUDA专有错误，也不等于旧CPU/NPU GDN xfail已解释或解除。
尚未确定首个局部伴随/舍入差异，不以scalar选择或原始失败很小宣称修复。

## 干净检出复现

使用相同CUDA/Torch私有环境；只取授权新分支，构建与运行目录均新建。无需旧运行材料。

```sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCH_CUDA_ARCH_LIST=8.6 NVIDIA_TF32_OVERRIDE=0
python scripts/build.py --backend cuda --build-dir NEW_CORE --jobs 56
PYTHONPATH=scripts python - <<'PY'
import subprocess, sys
from core5_resident_scope import COMPONENTS
subprocess.run([sys.executable, 'scripts/build_device_control.py', '--core-build', 'NEW_CORE',
    '--build-dir', 'NEW_RESIDENT_CPP', '--runtime', 'standalone', '--cuda-architectures', '86',
    '--jobs', '56', '--checks', *COMPONENTS], check=True)
PY
python scripts/build_device_control.py --core-build NEW_CORE --build-dir NEW_RESIDENT_PY --runtime python --python-component weighted --cuda-architectures 86 --jobs 56
python scripts/verify_core5_resident_single.py --device cuda:0 --core-build NEW_CORE --python-resident-build NEW_RESIDENT_PY --standalone-resident-build NEW_RESIDENT_CPP --output-dir NEW_GATE
python scripts/check_cuda_eager.py --device cuda:0 --clients libtorch --core-build NEW_CORE --dtype float16 --output-dir NEW_HALF
NEW_CORE/tidegraph-weighted-training-check --device cuda:0 --dtype float16 --scoring-dtype float32 --placement mixed-c --case feedback/ssd --packing scalar --output-dir NEW_SCALAR
python scripts/check_cuda_eager.py --device cuda:0 --peer-devices cuda:0 cuda:1 --clients python-native libtorch --core-build NEW_CORE --case gated-hb --cpp-case hb/gated_delta --output-dir NEW_PEER
```

长命令用[容器监督器或本机服务](../verification.md)包装；正式资格对不可变源码运行。
下一动作和活动作业只见[STATUS](../STATUS.md)，未解除限制只见[ROADMAP](../ROADMAP.md)。
