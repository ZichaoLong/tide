# RTX3090 core-5 eager、传输与首轮构建 — 2026-10-31

[逐配置记录、固定源码、命令和hash](core5-cuda-eager-20261031.json)。数学固定
`0770593838a37c790a568226930567ed60f99b83`。所有下表作业已终止；测试耗时不作为性能。

目标为Ubuntu22.04/x86_64、4×RTX3090/24GiB/SM86、驱动595.71.05、
Torch2.12.1+cu130/Python3.12.3、Toolkit13.0.88、GCC11.4。
使用项目私有venv，公共环境/驱动未修改。affinity112、CPU配额56、内存240GiB；
构建56线程，显式8.6/86。eager的ATen/BLAS/interop及native worker均1，packed，
FP32显式禁用TF32；FP16 payload/FP32 scoring、FP32 masters和原loss scale16。
完整原断言、AdamW、输入、epsilon保持；没有因新GPU失败改容差。

## 分客户端真实结果

固定实现/门禁/core：`c96296150e005363346eae46fcfed8613b3956d2`。
core共209个Ninja目标成功，Python native动态加载、standalone实际运行分别成立。

| 范围 | 纯PyTorch | Python native | standalone LibTorch |
| --- | --- | --- | --- |
| FP32 mixed-c | 19/19通过 | 19/19通过 | 23/24通过，整体gate失败 |
| FP16 mixed-c | 19/19通过 | 19/19通过 | 22/24通过，整体gate失败 |
| FP32 KV mixed-a/b | 2/2通过 | 2/2通过 | 2/2通过 |

Python19例覆盖五记忆、短卷积、异构attention、FC/HB/反馈PDG、非对齐DAG、一般端口及十种发送代表；
standalone24例为4拓扑×5记忆/异构。具体输入/配置由固定源码的
`weighted_device_cases.py`和`cpp/test/weighted_graph_fixture.cpp`生成，不能把不同fixture互相代替。
Python各例检查完整返回/控制/逐边输出/history/pending、独立descriptor/control/raw-state根、
streaming/prefill、推理、保留窗口、共享参数、四次更新和optimizer。
standalone此版本比较完整返回、全目标梯度、共享参数、保留窗口与四次更新；
后续6cacad0新增单输出/descriptor/control/state独立根检查，其正式资格另列。

同栈新owner的完整第四步返回、梯度、参数和optimizer要求逐值相等；CPU/重映射设备handoff仍用原容差。
这些小型机制检查不认证模型收敛、全图族/规模或所有组合；旧CPU/NPU失败及strict xfail不变。

## 原严格失败与隔离

- FP32 standalone `fc/ssd`：第3步AdamW后`regions.0.head.bias`，CPU
  `0.11622299998998642`，GPU `0.11622648686170578`，差`3.4868717e-6`，
  超原`atol=1e-6, rtol=1e-5`。该步梯度CPU`-1.4730905e-7`、GPU`-1.5774422e-7`；
  更早两步梯度差约1e-11。GPU scalar同参数仍失败；同输入CPU FP32/FP64通过。
  mixed-a另在恢复后`nodes.2.extra.kernel.conv.bias`出现2/8坐标逐值不等，最大`5.82e-11`。
  这些隔离不说明局部数学公式已定位或修复。
- FP16 standalone `feedback/affine`、`feedback/ssd`：原tensor比较失败，首轮日志未定位字段；
  新checker已增加字段、坐标、实际值和阈值诊断，等待固定来源后续记录。
- 双卡native `gated-hb`：初始完整比较和前三步通过，第四步精确恢复在
  `nodes.0.kernel.read_out`梯度3/4坐标不等，最大`4.44e-16`。保持失败，不能以很小差值改成通过。
  native按分配地址排序packing的独立反例已修复，相关CPU110项和C++两dtype通过；
  原双卡恢复失败仍相同，故该修复不解除恢复边界，也不解除旧workers2失败。

## 多卡可用路径

固定探针`31d0afde0eda537c90286ba3e27f955eabd74aed`，复用hash匹配的c962961 core。
四卡全部12方向P2P access/原生跨卡原子均0，enable均217。
72单元（12方向×普通Python/native/显式Python CPU桥×默认/非默认流）
各4096个FP32值、别名和独立None/零根逐值通过。

物理0/1的后续独立范围：

- transfer根/optimizer测试通过；standalone通用owner检查36例、72更新、144连接窗口通过。
- core-5普通复制：五记忆+异构的PyTorch6/6、native5/6；失败为上述GDN精确恢复。
- 显式Python CPU桥`delta-pdg`完整根、四更新、精确同栈/CPU/重映射恢复通过。
  这是诊断桥的明确配置，不是新增native主机中转协议。

Python探针和standalone generic出现AccumulateGrad流不匹配警告，原日志hash保留。
普通复制成功不能证明直接P2P，也不认证CUDA Graph capture、NCCL、多卡resident或性能。
当前多卡resident仍缺双向P2P和原子能力；结论仅针对本机/容器配置。

## 构建、作业机制与后续范围

`1fe1e379bec0d0d28e40be1dbfdfbe647add4171`修复CUDA13泛型graph-node新增edge-data参数：
standalone resident630目标、Python resident255目标成功，控制图仍为设备WHILE/SWITCH。
这不等于resident完整正确性或设备时间线通过。
原86-real配置失败、CUDA13旧API编译失败，以及首次control测试末尾要求CANN式workspace>0的失败均保留。
8a34f66的CUDA所有权/增量预算修订已定向开发检查，固定资格另列。

无systemd容器采用有超时的Linux subreaper，TERM→KILL清理所有后代组；
实际SSH断连后任务父子仍存活，stop后cancelled且无运行后代。
14项目标启动器/架构检查及小型CPU公式92项属于开发检查，不冒充完整x86 CPU资格。

## 干净检出复现

只clone授权新分支，在相同私有Torch/CUDA环境执行；路径均为新输出，既有原始运行目录非必要输入。

```sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCH_CUDA_ARCH_LIST=8.6 NVIDIA_TF32_OVERRIDE=0
python scripts/build.py --backend cuda --build-dir NEW_CORE --jobs 56
python scripts/check_cuda_eager.py --device cuda:0 --clients pytorch python-native libtorch --core-build NEW_CORE --dtype float32 --output-dir NEW_FP32
python scripts/check_cuda_eager.py --device cuda:0 --clients pytorch python-native libtorch --core-build NEW_CORE --dtype float16 --output-dir NEW_FP16
python scripts/check_cuda_peer.py --devices 0 1 2 3 --native-library NEW_CORE --output-dir NEW_PEER
python scripts/check_weighted_device.py --device cuda:0 --peer-devices cuda:0 cuda:1 --implementation native --native-library NEW_CORE --placement mixed-c --scoring-dtype float32 --packing packed --workers 1 --case gated-hb --output-dir NEW_GDN
python scripts/diagnose_weighted_async.py --device cuda:0 --peer-device cuda:1 --copy-policy cpu_bridge --scope complete --case delta-pdg --output-dir NEW_BRIDGE
```

历史原断言用上述固定提交重新构建；当前新checker包含更强独立根，不声称与首轮测试库存相同。
实际job包装、停止命令见[verification](../verification.md)，下一步与活动作业仅见[STATUS](../STATUS.md)。
