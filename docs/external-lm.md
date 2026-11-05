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
默认加载前检查实际空闲HBM，预留max(resources.device_reserve_gib,总量10%)（默认6.4GiB），CPU也检查主存；没有自动OOM搜索、降精度或缩batch。
预算是保守估算，不是实测峰值。layer_split按层连续分配、embedding/head在逻辑设备0，
最后归一化在末设备；不整除卡数也可用。它使用Accelerate普通跨层复制，**不是TP/EP/HCCL资格**。
公共接口可描述CUDA，但当前有限计划仅CPU/NPU，未恢复已释放的gpu3090。

## 显式容量试跑与单配置后台运行

`execution.device_memory_admission`省略时为`strict`，旧配置内容和身份保持不变。
显式`observe`适用于CUDA/NPU的Transformers推理：仍计算原逐卡预算、读取实际free/total，
保留原reserve，超限时打印警告并尝试分配。它只放行静态设备内存估算；主存、磁盘、设备数量、
模型资产、空闲卡队列、超时、非有限、OOM和其他运行异常均保持原处理。
它不把设备空闲内存改成另一个值，不改变dtype、batch、输入、缓存长度或算子。

[双NPU B512显式试跑](../examples/external_lm/qwenmoe-two-npu-b512-observe.json)保持原
Qwen3-30B-A3B-Instruct-2507、BF16、L512/O128、prefill_chunk64、16线程、一次warmup和一次正式生成，
只增加observe身份，并将整流程计算上限明确设为7200秒。原准入失败及所有数值失败仍保留。
`lm plan`输出生效策略；`result.json`在模型分配前落盘`device_admission`，包括逐卡estimate/free/reserve、
within_budget及decision。即使随后分配失败，这些准入观察仍可查；若尚未构造成功，实际峰值可能尚未采集。
终端/CSV/HTML均显示memory admission，成功完成也只代表容量探索执行，独立数值/恢复另验。

`lm queue --config CONFIG`把一个JSON配置展开为单项、有身份的清单，使用原队列和进程监督，
与`--manifest`互斥。它保持配置线程，不加入旧矩阵、不自动重试。公共前台`lm run`不会自行提供
持久服务或实施配置中的时限；本机长试跑按下例提交。先按上文准备环境/资产，在固定干净检出中
完成安装和相关资格；`FIXED_SOURCE`必须指向该固定检出，不能是继续编辑的工作区。

```sh
lm_root=/absolute/experiment-root
lm_src=/absolute/fixed-source
lm_attempt=qwenmoe-b512-$(date +%Y%m%d-%H%M%S)
lm_helper=/absolute/use-local-ascend/scripts/run_on_free_npu.py
PYTHONPATH="$lm_src/python" "$lm_root/env/transformers/bin/python" -B "$lm_src/scripts/submit_lm_job.py" \
  --unit "gef-lm-$lm_attempt" --output "$lm_root/jobs/$lm_attempt" --seconds 32400 --memory-gib 384 -- \
  bash "$lm_src/scripts/run_lm_local.sh" "$lm_root" queue --root "$lm_root" --backend npu \
  --config "$lm_src/examples/external_lm/qwenmoe-two-npu-b512-observe.json" \
  --npu-helper "$lm_helper" --attempt "$lm_attempt" --hours 9

# 调度状态；模型分配后，尾随日志中的warning、OOM及完整traceback：
cat "$lm_root/runs/npu-$lm_attempt/queue.json"
tail -F "$lm_root/runs/npu-$lm_attempt/manual-qwenmoe-npu2-b512-l512-observe/process/task.log"
# 结果（只有内部execution_status=completed才算执行完成）：
tidegraph lm results "$lm_root/runs/npu-$lm_attempt" --format table
# 停止后检查完整服务进程组及持久状态：
systemctl --user stop "gef-lm-$lm_attempt.service"
systemctl --user show "gef-lm-$lm_attempt.service" -p MainPID -p ControlGroup
cat "$lm_root/jobs/$lm_attempt/status.json"
```

空闲卡选择及项目计算锁等待各自有6小时时限；每个计算过程上限2小时，整个服务先到9小时则停止。
该队列自动选择两张无其他进程的卡，物理/逻辑映射记录在case的allocation.json；不手填已占用卡号。
`observe`代表用户有意进行的容量探索，不能从静态估算56.72GiB推断B512实际峰值或成功率。

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
模型/恢复子进程的cwd固定为各case的`process/work`或`restore-process/work`，输入、输出与资产
使用绝对路径。厂商相对诊断留在工作目录；延迟提交继续严格要求源码干净，不忽略或删除污染文件。
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
及`--retries 0..32`、`--io-timeout 1..180`秒（默认60）；原asset-state按内容身份保留于attempts，当前状态记录偏移和重试原因，不保存认证URL。
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

## 11-04明确授权的失败重启

`tidegraph lm retry-matrix --group all|compute|llama|placement`从跟踪生成器重建21项，
不依赖旧运行目录：19个未执行依赖点保持完整原配置；OLMoE CPU B512将整流程上限由1200秒
显式改为7200秒；原单NPU B512估算63.080GiB仍拒绝，新增双NPU显式跨层配置每卡约36.8GiB。
后者保持B512、输入512/输出128、BF16和prefill_chunk64，是新放置配置，不称原单卡通过。
已完成的曲线、检查和数值失败不重新排队；原OLMoE双NPU小检查/精确恢复按原身份复用。

`scripts/dispatch_lm_remaining.py --manifest COMPUTE_JSON --small-job NEW --moe-job NEW`
按资产分别提交11项选择性队列；`scripts/run_lm_engine_followup.sh ROOT --kind llama|placement
--manifest SELECTED_JSON --attempt NEW --small-job NEW --moe-job NEW`运行其余6/4项。
二者均可用`--npu-helper`指定已审阅的站点调度入口，保持整组锁和两次资源采样。
llama加`--network-job ''`允许经独立试测可用的直连源码下载，仍由compute.lock隔离构建和计时。

模型下载复用原hash/断点，`fetch_lm_remaining.py --chunk-mib 8 --retries 10 --attempt NEW`
明确缩短Range并给每文件10次临时错误重试预算；第二个模型服务以`--wait-for-job FIRST_JOB`
等待第一个终态，避免两个大型模型争用同一路代理。依赖等待最多30小时、服务总时限由提交记录约束。
源码归档较小且不支持Range，直连重新取得完整固定归档；旧失败日志和partial保留。
下载网络不改变共享路由、认证或公共软件，错误hash/Range/非有限/通信/精确恢复失败仍不可豁免。

若用户已明确允许低占用卡共享，先用`scripts/run_shared_npu.py`的`tide-npu-sharing-v1`
策略绑定实际PID/start/UID、HBM/利用率/空闲余量和失效时间，然后用
`scripts/make_lm_npu_adapter.py --policy POLICY --helper SITE_HELPER --output NEW_ADAPTER`
生成`--npu-helper`入口。策略与helper都按hash固定；新增占用进程、超过预算或过期均拒绝，
每case保存sharing-audit.jsonl。未列入共享策略的卡仍要求无进程；不称共享观察为独占计时。


## cwd隔离修复后的选择性后继

`lm retry-matrix --round isolation --group all|compute|llama|placement`生成15项：
Qwen7B CPU/NPU两项、Qwen30B-A3B CPU/双NPU八项、四NPU小检查及两个成本点，
和两个llama B32后继。前13项完整配置不变；两个B32仅将整流程上限1200→7200秒并改名，
不重跑已完成的OLMoE B512、llama B1及原独立数值失败。原始21项仍可用`--round initial`重建。

本轮下载只续传Qwen MoE，c28单流、32MiB Range、每次连接/读取120秒、每文件24次临时重试；
外层服务30小时，项目160GiB/账户300GiB停止线保持。具体入口：

```sh
tidegraph lm retry-matrix --round isolation --group compute > compute.json
tidegraph lm retry-matrix --round isolation --group llama > llama.json
tidegraph lm retry-matrix --round isolation --group placement > placement.json
# 在固定源码的持久服务中执行，ROOT和输出路径须为绝对路径：
python -B scripts/fetch_lm_remaining.py ROOT --group qwen-moe --chunk-mib 32 --retries 24 --io-timeout 120 --attempt UNIQUE
python -B scripts/dispatch_lm_remaining.py ROOT --attempt UNIQUE --manifest /absolute/compute.json --small-job assets-small-g --moe-job NEW_MOE_JOB
# 启动前将现有engines/llama/ready.json作为只读小文件保存至PREVIOUS_BUILD_JSON。
bash scripts/run_lm_engine_followup.sh ROOT --kind llama --attempt UNIQUE --manifest /absolute/llama.json --small-job assets-small-g --moe-job NEW_MOE_JOB --network-job '' --previous-build PREVIOUS_BUILD_JSON
```

llama新增construction/warmup/measurement/prefill/decode/checkpoint阶段日志；有限频率的日志写入
包括在各阶段时间内。即使超时，完整阶段记录及截断末行仍保留，不从部分进度估计完整吞吐。
构建以C++客户端文件hash区分目录，旧binary/shared libraries及build record继续保留；
已有checkpoint恢复使用原记录的binary/GGUF身份，不随新的ready指针切换。
`--previous-build`在成本点之前用B2/L16/O4对旧/新二进制检查tokens、logits、完整序列化状态及
各自新进程恢复；失败阻止该后继。此检查只验证阶段记录改动，不解除旧独立数值失败。

若旧运行目录已退役，从79197ad固定检出按本页安装/构建流程重建原客户端，保留其ready.json，
再从当前固定检出在同一新实验根准备后继；无需找回历史checkpoint或性能输出。
输入由固定生成器重建，模型资产仍必须校验发布hash。新独立重建不能冒称原机器二进制或旧测量。

本轮15项的实际终态与成本见[11-05审阅](evidence/llm-isolation-terminal-20261105.md)：
13项成本完成、1项严格数值/token失败、1项双NPU B512容量拒绝；配置生成入口不表示全部资格通过。
