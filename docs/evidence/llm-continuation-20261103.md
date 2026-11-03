# Dense/MoE后继后台提交与初步诊断（2026-11-03）

用户授权“完成剩余事项，放到后台执行”。本报告验收实现、固定设施与实际提交，
并审阅已结束的小诊断；仍在运行/排队的模型成本不记通过。完整快照见[JSON](llm-continuation-20261103.json)。

| 固定源码 | 已验事实 | 尚未据此证明 |
| --- | --- | --- |
| 8b45cb42325224e38f945405988f5efe0bd83e3e | 分段下载相关27项；真实Range；两路持续推进 | 未完成分片的最终hash、MoE执行 |
| 472d94ffdcb23f6dc270626cd7089945d2a3b833 | 新wheel安装与固定94项设施检查；选择性清单、dispatcher实际启动 | 43项全部终态/通过 |
| 983da722038814d24c03659c7187ec769b28b263 | 固定48项、固定llama头文件C++语法检查；两服务实际启动 | llama完整构建、GGUF、实际模型和精确恢复 |

94和48项存在重叠，不累计成独立模型资格。旧80项矩阵及失败保持[原身份](llm-terminal-audit-20261103.md)。

## 后继范围与执行

- `lm recovery-matrix`：32个下载依赖点原配置不变；4个CPU原1200秒超时点换新名字，显式7200秒整流程上限，模型/输入/训练配方不变；7个小诊断。总43项。
- ready CPU/NPU队列已运行。small、moe组各等自己的下载终态，再由dispatcher提交独立24小时CPU/NPU服务。
- `lm engine-matrix`：4个Qwen1.5B/OLMoE B1/B32独立CPU llama.cpp点；6个NPU显式跨层后备点及4个小检查。
  后备是Qwen1.5B/OLMoE双设备B1/B32、Qwen MoE四设备B32/B128。不是vLLM或TP/EP。
- llama服务等small下载终态再下载源码，确保至多两路大下载；构建/转换持有compute.lock，正式计时不与重型构建重叠。
  构建按实际affinity，计时16线程；依赖30小时、服务42小时、单点1200秒、锁/设备等待6小时。
- 所有7个活跃service均核对为background.slice、独立ControlGroup。当前主仓库可编辑，外部固定源码不能修改。
  新服务未做主动停止演练，沿用原已验证的同一子reaper/lifecycle；不能把请求MemoryMax当v1逐unit硬隔离。

## 已完成的小诊断

固定原atol=1e-5/rtol=1e-4，每条greedy轨迹独立，没有参考token指导候选。

| 诊断 | 观察 |
| --- | --- |
| CPU FP32同设备eager完整前缀 vs cached，0.5B/1.5B | 首超差均在logits.1，输出token相同；末层KV另完整比较 |
| 同eager前缀的全部/最后logit投影 | 两模型、两平台小材料均通过；不能据此将所有线性投影舍入排除 |
| CPU FP32同cached eager vs SDPA | 两模型logits.0即超差、token相同 |
| NPU BF16同cached eager vs SDPA | 两模型均数值和token不同；改变路由/工作量不得称严格等价加速 |
| 同eager前缀 CPU FP32 vs NPU BF16 | 两模型数值和token不同；同时含跨设备/精度因素，不能只归因BF16 |
| 0.5B FP32/eager手写CE、完整梯度、第一步Adam方程/槽 | CPU新配置通过，NPU仍失败；不修饰旧BF16检查失败 |

NPU原L2048/micro4静态73.23GiB拒绝保留。当前torch_npu的SDPA flags说明只设置全局变量，
不能证明实际融合路径。新增micro1保存张量、dispatch与峰值诊断在后台；不按峰值简单乘4撤掉保护。
自身恢复不替代独立数值，超差成本不称严格资格；非有限和通信错误仍失败。

## 第二实现的边界与有界停止

llama.cpp固定`2cbca389f515df455412dafa07e2b1359cc72a6d`。C++消费相同独立token材料，
显式BF16 KV、GGUF MOSTLY_BF16保留FP32 1D张量，native ARM/OpenMP，无量化/精度后备。
与HF全BF16不能作完全同精度的等价加速比。小检查覆盖完整logits/token，独立KV尚不可读，明确not_checked；
B1的新进程恢复另比较完整序列化状态与下一步logits。构造、warmup、保存/恢复与计时分列。

vLLM-Ascend审阅固定`c09d3e20b9c45c80b34ae3781440d90904a44023`：
[官方安装](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/docs/source/getting_started/installation/install_vllm_ascend.inc.md)
明确普通pip wheel面向A2，A3需要源构建/变体；
[现有CANN入口](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/docs/source/getting_started/installation/existing_cann_environment.inc.md)
要求NNAL set_env/libatb；
[requirements](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/requirements.txt)
固定Torch2.10/TorchNPU2.10.post7、Transformers5.14.1。
公共基线2.12/CANN9.1发布记录明确没有NNAL，两个惯用NNAL路径实查也不存在。
本轮不安装这套额外原生栈，执行已获准的跨层有限后备；没有尝试A2 wheel、降级公共库或声称A3硬件不支持。
重启vLLM/TP/EP需要独立A3 NNAL/源码配套，先小collective、算子/模型/恢复，再有限B32/B128；
本次显式复制不能代替该资格。

精确命令、unit和日志见唯一[STATUS](../STATUS.md)，公共配置和操作见[external-lm](../external-lm.md)。
