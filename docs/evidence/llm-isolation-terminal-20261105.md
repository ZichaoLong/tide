# Dense/MoE目录隔离后继终态审阅（2026-11-05）

固定实现`694eb605b6769f8d3f208710a920dbfbf504abf4`的15项后继已全部终态。
最后一个服务于北京时间2026-11-05 05:43:00结束；9个外层服务inactive、MainPID=0、ControlGroup为空，
9份job与15份case进程记录无残留后代，48个PID/start身份实查无匹配存活进程。
本次只核对现成结果、文件身份及进程，未重启、延长或新增模型工作。
用户要求中断时修复；本批没有中断，容量拒绝与严格数值失败分别保留。
[启动证据](llm-isolation-launch-20261104.md)不改写；逐项配置、原结果hash、工作量、峰值与失败见
[15项JSON](llm-isolation-terminal-20261105.json)。原21项失败仍见[前次终态](llm-restart-terminal-20261104.md)。

## 实际完成范围

| 结果 | 数量 | 含义 |
| --- | ---: | --- |
| 推理成本测量完成 | 13 | numeric/discrete/checkpoint均not_checked，独立资格未建立 |
| 独立小检查执行完成 | 1 | Qwen MoE四NPU，numeric/discrete均失败，未选择checkpoint |
| 模型分配前容量拒绝 | 1 | Qwen MoE双NPU B512，未发生模型OOM、未产生吞吐 |
| 中断、超时、依赖未跑、待排队 | 0 | 不能由9个外层exit0覆盖上述两类失败 |

14项执行完成不等于14项严格通过。固定设施125项通过和旧/新llama小回归仍只覆盖其原范围；
本次15项没有新增checkpoint恢复检查，不继承此前B1或小回归的精确恢复资格。

## 正式成本观察

全部输出128、16线程，一次独立流程和一次正式测量。Transformers为BF16、SDPA、TF32关闭；
llama为MOSTLY_BF16 GGUF（1D张量FP32）及BF16 KV，ARM/OpenMP、BLAS关闭。
正式时间含输入上传、在线调度、计算、有限值检查、token/统计返回与同步；构造/warmup另列于JSON。
未排除其他项目争用；项目计算锁使本项目构建与正式计时分离。

| 模型 / 客户端与放置 | B | 输入长度 | 正式秒数 | 输出token/s |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B / Transformers CPU | 8 | 512 | 176.617 | 5.798 |
| Qwen2.5-7B / Transformers 1×NPU | 8 | 512 | 3.736 | 274.072 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers CPU | 1 | 512 | 235.407 | 0.544 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers CPU | 8 | 512 | 368.626 | 2.778 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers 2×NPU | 1 | 512 | 62.153 | 2.059 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers 2×NPU | 8 | 512 | 105.527 | 9.704 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers 2×NPU | 32 | 512 | 171.293 | 23.912 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers 2×NPU | 128 | 512 | 308.329 | 53.138 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers 2×NPU | 8 | 2048 | 163.204 | 6.274 |
| Qwen2.5-1.5B / llama.cpp CPU | 32 | 512 | 1332.666 | 3.074 |
| OLMoE / llama.cpp CPU | 32 | 512 | 1042.777 | 3.928 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers 4×NPU | 32 | 512 | 192.189 | 21.312 |
| Qwen3-30B-A3B-Instruct-2507 / Transformers 4×NPU | 128 | 512 | 377.007 | 43.458 |

Qwen MoE实际参数量30,532,122,624，固定模型为`Qwen/Qwen3-30B-A3B-Instruct-2507`，
revision=`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`。双卡参数分别15,577,225,216和14,954,897,408；
四卡分别8,099,777,536、7,477,447,680、7,477,447,680、7,477,449,728，总量相同。
这是Accelerate跨层显式复制，不是TP/EP或vLLM。四卡B32/B128在本次观察中均慢于双卡，
不能由增加卡数推断加速；数值/生成token资格未建立，也不称严格等价加速或减速。

B128/L512实际prefill为8次×128样例×64时间，每次8192输入行；decode127次×128样例。
L2048为32次64时间批。样例来自不同的受控合成token序列，本轮这些点长度相同、全活动，
并不证明变长动态服务能力。没有采集本次内部MM/BMM形状或专家桶，客户端行数不证明专家充分合批。

Qwen MoE双卡B128 allocated峰值34.078/32.880GiB，reserved峰值60.883/60.652GiB；
四卡B128 allocated为18.276/17.078/17.077/17.078GiB，reserved为60.555/60.893/60.869/60.893GiB。
参数分摊降低allocated，但reserved仍接近卡容量；不能按allocated直接推大batch余量。
各点真实物理卡、构造、warmup、RSS及allocated/reserved均保留在JSON。

## 两类失败与停止条件

四NPU小检查B2/L16/O4执行完成。独立CPU FP32全前缀调度与BF16在线缓存候选在
4组logits及96组KV字段比较全部超原atol1e-5/rtol1e-4；生成8个token中2个不同。
首组logits最大绝对差1.085041，末组34.137028；后续KV差异可能同时受已分歧token影响，
现有记录不能把全部差异归因于舍入或通信。独立路由负载未检查，不将成本称等价工作量。
原阈值不改，重启数值修复须先沿首个分歧定位，而不是重复相同全图或放宽容差。

双NPU B512/L512在模型分配前拒绝：逻辑device0估算56.719GiB，实查空闲61.137GiB，
保留6.4GiB后可用54.737GiB，缺1.983GiB。没有模型执行、OOM或中断；这是当前配置的静态准入边界，
不宣称不可突破的硬件上限。后继应先拆解KV/激活/workspace及allocator峰值，
比较显式更小prefill时间批或更多设备的新预算；不缩小逻辑B、不降低安全余量或直接重复原配置。
本次没有自动追加四卡B512，也不由四卡B128外推其容量。

## 下载、隔离与保留

Qwen MoE16/16权重分片及全部配套文件在04:46:51完成发布hash校验，共61,084,262,963 bytes。
本次复查记录与实际size/mtime一致，没有再扫描61GB权重hash。
单流32MiB Range、I/O120秒、每文件24次临时重试在两次读超时后成功；旧下载失败仍保留。
固定检出仍干净，厂商fusion_result.json写在case的process/work，延迟提交后续队列成功，
由真实本机NPU过程补充了目录隔离回归的实际证据。

账户quota约176GiB、实验根du约122.54GiB，低于账户300/项目160GiB停止线与800GiB硬限。
未清理权重、旧二进制、checkpoint或失败记录；无需GPU3090，不存在待运行本批服务。
有限清单可由已安装公共入口重建：

```sh
tidegraph lm retry-matrix --round isolation --group all > isolation.json
# 仅查看已有结果，不启动模型：
tidegraph lm results /absolute/lm-root/runs --model qwenmoe --format csv
```

完整安装、配置、预算与运行入口见[external-lm](../external-lm.md)。当前接续只维护[STATUS](../STATUS.md)，
剩余边界及重启条件只维护[ROADMAP](../ROADMAP.md)；不把本报告变为新的排队清单。
