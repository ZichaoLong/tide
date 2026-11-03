# 外部语言模型：可安装入口与后台矩阵

本入口测量外部预训练语言模型，独立于core-5图执行器及其资格。
固定模型与已授权上限见[准备报告](evidence/llm-experiment-preparation-20261102.md)；
该历史报告仍保留“当时未授权”的事实，当前执行状态以[STATUS](STATUS.md)为准。
core-5样例窗/s与LM token/s不直接相除；参数量相近也不表示相同函数、模型效果或工作量。

## 安装、配置与单次操作

Python>=3.10。项目私有环境安装固定tidegraph wheel；Transformers4.57.1、Accelerate1.11.0、
PEFT0.17.1、tokenizers0.22.1。本机复用只读公共Torch2.12/TorchNPU2.12/CANN9.1，
用`scripts/prepare_lm_environment.sh`配置增量；不要把普通PyPI vLLM强装到公共基线。
在另一平台选择该平台匹配的Torch，而非复制本机NPU环境。

```sh
tidegraph lm --help
tidegraph lm catalog
tidegraph lm matrix > finite-matrix.json
tidegraph lm plan examples/external_lm/qwen15-cpu.json
# 单模型、有界分段尝试；root为共享模型缓存，project-root计入所有环境/模型/结果。
tidegraph lm fetch qwen15 --root /absolute/experiment/models --project-root /absolute/experiment
# 前台单次入口；长任务须放持久服务，由队列实施配置的timeout_seconds。
tidegraph lm run examples/external_lm/qwen15-cpu.json --assets /absolute/experiment/models --output /absolute/experiment/one-run
# 只用于此前显式checkpoint=true的run；新进程、新owner核对下一步。
tidegraph lm resume /absolute/experiment/one-run --assets /absolute/experiment/models
# 查询无需导入Torch、不启动实验；CSV/HTML可直接查看。
tidegraph lm results /absolute/experiment/runs --format html --output /absolute/experiment/results.html
tidegraph lm results /absolute/experiment/runs --model qwen15 --backend npu --format csv
```

配置分model、task、execution、measurement、resources；JSON未知字段被拒绝。
[CPU推理](../examples/external_lm/qwen15-cpu.json)、[双NPU MoE](../examples/external_lm/qwenmoe-two-npu.json)、
[LoRA训练/恢复](../examples/external_lm/qwen15-lora-npu.json)可复制修改。
plan不分配模型，输出完整配置身份、固定revision、权重/KV/激活/Attention/workspace及逐设备预算。
加载前检查实际空闲HBM，预留至少max(6.4GiB,10%)，CPU也检查主存；没有自动OOM搜索、降精度或缩batch。
预算是保守估算，不是实测峰值。layer_split按层连续分配、embedding/head在逻辑设备0，
最后归一化在末设备；不整除卡数也可用。它使用Accelerate普通跨层复制，**不是TP/EP/HCCL资格**。
公共接口可描述CUDA，但当前有限计划仅CPU/NPU，未恢复已释放的gpu3090。

## 计时、输入和检查契约

主矩阵固定输出128token、greedy、忽略EOS以控制工作量；每条样例独立seed/token。
变长组使用项目生成的多主题文字和独立编号/数值，长度覆盖1/4、1/2、完整长度，允许自然EOS。
它不是外部自然语料或模型效果评价。完整生成token保留；缓存留在模型设备，
无需每步下载logits/KV。记录真实token、逻辑batch、prefill块长、物理行、活动行、padding和实际参数位置。
prefill_chunk只切时间，不切逻辑batch；KV完整保留，后续块仍看到全部合法历史。
当前变长decode不做物理batch压缩，结束样例占填充行；不得用有效token数冒称物理计算等量减少。

计时包含输入上传、在线推进、算子、有限值检查、返回token/统计及同步；训练另含loss、
反向、梯度有限值检查、AdamW、参数/槽检查与清理。prefill与decode分开，并单列构造、warmup、
checkpoint和新进程恢复。synthetic token生成在计时前；同一静态材料不含参考轨迹。
峰值覆盖整个进程，含构造/warmup/恢复准备，不能误称纯计时窗峰值。
profile独立进程，CPU/CUDA保存实际算子形状/调用，NPU用厂商profiler；另观察MoE gate top-k负载。
profile耗时不进入正式吞吐。engine支持Transformers及下述有界独立CPU llama_cpp；其他名称明确拒绝。

训练是新固定配方：global batch4、micro1/4、AdamW lr1e-4/betas(.9,.999)/eps1e-8/wd0、
非padding移位CE均值；各micro等长，按完整global batch归一化。全参FP32参数/梯度/槽，
显式BF16 autocast；LoRA冻结BF16基座/router，仅Attention Q/V的r8/alpha16/dropout0适配器可训练，
adapter/槽FP32。OLMoE辅助loss系数0.01。1次warmup更新后3次计时更新；warmup改变模型状态，已记录。
FP16训练未实现loss scaling，本配方提前拒绝；不把LoRA当全专家训练。

`lm check CONFIG --assets ROOT --output NEW`要求显式B<=2/L<=32/输出<=8。
CPU FP32参考每步完整前缀重算，候选独立online cached，比较全部logits、层KV及输出token。
固定atol1e-5/rtol1e-4，不随dtype放宽。BF16与FP32超差可以作为差异观察，不变成通过。
`lm training-check`另用Qwen0.5B B1/L<=32，比对手写CPU CE、完整参数梯度及第一步AdamW方程/槽。
严格数值、离散token、路由（尚未独立核对）、执行、恢复分别记录；小锚点不外推大点。
`lm resume`检查缓存/全部可训练参数/Adam槽在新进程中的精确下一步，不代替独立数值比较。
非有限、错误通信、非法配置、容量与超时是失败，不被探索模式吞掉。

## 有界队列与源码隔离

`lm matrix`生成71个可执行测量/观察配置（44主曲线、4变长、14训练、3线程、2精度、4profile），
另有9个小型数值/双设备放置检查。10个优化引擎/TP、EP点在manifest明确列为尚未排队；
需要配套引擎与真实通信检查，不能映射成另一遍Transformers冒充。

本机使用固定干净worktree、非editable wheel、仓库外环境与模型资产。
`scripts/submit_lm_job.py`复用公共子进程监督器，在background.slice提交独立service：
首批下载、环境准备、CPU队列、NPU队列、条件后批下载分别有状态与日志。
CPU/NPU正式计算共用项目flock；队列等待模型时不占NPU，设备准入沿用两次采样、整组锁、无其他进程。
第一阶段与后批分开；后批下载只在首批下载终态且Qwen1.5B有完成的B1校准后启动。
后批资产完成后自动提交独立第二阶段CPU/NPU服务，不让前一队列的24小时时限覆盖长下载之后的计算。
每个计算/恢复过程1200秒，设备/项目锁等待最多6小时，计算service24小时，下载service30小时。
用`queue --attempt NEW`选择新输出身份；依赖失败和超时原样保留，不自动重跑配置。外层队列完成只代表清单处理完，不代表全部内部通过。

11-03授权后继由`lm recovery-matrix --group ready|small|moe|all`生成明确的剩余清单：
32个下载依赖点保持原配置，4个CPU超时点只将整流程上限显式扩为7200秒、输入/配方不变；
另加4个小推理因素诊断、2个FP32/eager训练检查及1个NPU micro1显存诊断。
`queue --manifest FILE --attempt NEW`验证完整展开配置、独特安全名称和清单hash，
严格使用选择的case和线程，不重新加入旧矩阵或隐式覆盖旧输出。
`scripts/dispatch_lm_remaining.py`依赖等待最多30小时；各组资产完成或下载终态后，分别提交
新的24小时CPU/NPU服务。ready组不等下载。下载失败项仍写dependency_failed，外层提交成功不代表计算通过。

`lm diagnose`只接受小推理配置，分别比较同dtype/设备eager完整前缀与缓存、完整/最后logit投影、
eager/SDPA及CPU FP32锚点。每条greedy轨迹独立，不把参考token喂给候选；记录首数值分歧与top2边距。
`lm memory-check`只对原NPU Qwen0.5B L2048/micro1单更新记录实际dispatch、保存张量和峰值。
它不撤销原micro4的73.23GiB保守准入拒绝，也不把micro1线性外推当安全证明。
诊断不计入吞吐；原atol/rtol、原失败及精确恢复身份保持。

```sh
# 具体unit与固定目录在STATUS；所有活跃目录都在主工作区之外。
systemctl --user list-units 'gef-lm-*' --all
python -B scripts/status.py
# 停止某个明确unit后，再核对其status.json的remaining_live_pids和实际ControlGroup。
systemctl --user stop gef-lm-cpu-20261102-a.service
```

用户可继续修改/提交主仓库；不要编辑/删除活动固定worktree、模型partial或私有环境。
正式测量期间不要并行启动本项目重型构建。停止条件后不会自动唤醒AI；之后按STATUS检查持久记录。
下载按公开SHA256/Git blob验证。默认32MiB Range分段，严格核对起止/总长、Content-Length、
identity编码；小型整文件可接受200。每文件最多6次临时连接/EOF/408、429、5xx重试（有限退避），
错误Range、hash、权限或预算直接失败；不能将重试变成完整性豁免。可显式设`--chunk-mib 1..64`
及`--retries 0..10`；原asset-state按内容身份保留于attempts，当前状态记录偏移和重试原因，不保存认证URL。
失败保留partial，服务终态后须明确重启才开启下一次预算。`scripts/fetch_lm_remaining.py`
分别处理OLMoE+7B及30B MoE两组，不重复已完成模型，一组内失败不阻止其他独立模型。
不输出netpath认证环境，不修改共享网络路线。项目首批80GiB/全阶段160GiB，账户300GiB停止新增大写入，
远低于800GiB硬限；tmpfs未作为必需依赖。

实现导航：`lm/config.py`预算和身份，`inputs.py`材料，`runtime.py`设备/参数位置，
`inference.py`在线缓存，`training.py`固定配方，`verification.py`/`training_check.py`独立锚点，
`checkpoint.py`新进程恢复，`matrix.py`有限清单，`queue.py`依赖/进程边界，`report.py`复用离线HTML。
修改配置/返回/配方时同步本页、示例、对应测试及计划身份；不改变core-5数学和旧资格。

## 有界第二实现和多设备后备对照

`lm engine-matrix --kind llama|placement|all`生成4个CPU llama.cpp成本点、6个NPU跨层放置点，
另有4个小检查。CPU为Qwen1.5B/OLMoE的B1/B32；NPU为这两模型双设备B1/B32，及Qwen MoE四设备B32/B128。
小检查执行异常会阻止同模型后续成本点；有限数值超差保留并允许显式成本观察。
与原单设备/双设备Transformers材料一致，输出128token、输入512、16线程；不重复已完成基线。

llama.cpp固定`2cbca389f515df455412dafa07e2b1359cc72a6d`，从官方固定归档构建公共C API消费者
[CMake](../examples/external_lm/llama/CMakeLists.txt)/[C++](../examples/external_lm/llama/main.cpp)。
`scripts/prepare_lm_llama.py ROOT`建立私有CPU构建与转换overlay；`--convert qwen15|olmoe`生成固定GGUF，
保留archive/binary/GGUF hash及转换身份。构建按实际CPU affinity，GGML native ARM/OpenMP、BLAS关闭；
正式计时线程16。独立C++过程不依赖Python/LibTorch，不称standalone LibTorch。
调用例为[配置](../examples/external_lm/qwen15-llama-cpu.json)配合公共`lm plan/run/check/resume/results`。

该客户端显式请求BF16 KV，失败不降为F16。GGUF MOSTLY_BF16仍保留1D FP32张量、GGML自身算术，
不能与HF全BF16当作完全相同精度的严格加速比。`check`核对独立CPU FP32完整logits/token；
GGML未暴露独立解码KV，标not_checked。B1另在新C++进程核对完整序列化状态和下一步logits逐字节续接。
计时含batch构造、算子、完整logit有限性归约和返回token；模型/context预分配、warmup、文件持久化另列。
n_batch/n_ubatch是实际配置容量，不冒充MM形状或证明内部没有拆分。没有模型效果或训练声明。

本机A3的vLLM-Ascend v0.30固定源`c09d3e20b9c45c80b34ae3781440d90904a44023`官方安装文档
要求A3源构建或对应变体、Torch2.10/TorchNPU2.10.post7及NNAL；普通pip wheel当前面向A2。
公共基线为2.12且明确未安装NNAL。本轮选择已授权的有限Transformers/Accelerate跨层后备，
[四设备配置](../examples/external_lm/qwenmoe-four-npu.json)不称vLLM、TP/EP或collective资格。
这不是硬件不支持vLLM的结论；重启引擎候选需单独匹配A3 NNAL与源码构建环境，不降级公共基线。

`scripts/run_lm_engine_followup.sh ROOT --kind llama|placement --attempt UNIQUE`为持久服务payload：
llama等待small资产服务终态后才占第二个网络流，构建/转换分别最多2小时/1小时并持有compute.lock，
转换后才逐模型提交小检查和有限点。依赖等待30小时，服务42小时；单配置仍1200秒、锁/设备等待6小时。
placement分组不等构建；有独立CPU/NPU队列资源协调。运行用户可继续编辑主仓库，固定检出与转换产物不得修改。
