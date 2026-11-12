# TL-REQ-006 — NPU 总占用高而 AICore 低：HB 训练的执行成本归因

2026-11-12。类型：性能容量 / 测量与诊断。状态：**待基座评审；尚无本需求的优化交付或下游验收。**
提出方：`ZichaoLong/tide` 的 `tide-learning-lab` 分支，本文件是权威请求；
基座目录中的 `docs/tide-learning-lab-TL-REQ-006-handoff.md` 是交接入口。
请求版本由包含本文件的提交固定，原始观测固定在下述 `89959587`，不会随新训练状态变化。

## 问题与实验目的

用户要求特别关注 AICore、AIVector、AICPU：总 NPU 利用率高不代表矩阵计算单元充分利用。
本批自然语言训练实测支持这一提醒：HB88 pooled 的总利用率中位数约94.5%，
AICore 只有3%，AICPU 为30–31%；入口树也有类似现象。
希望基座在后续性能工作中分辨必要数学工作、布局/反向/同步开销和硬件执行路径，
交付能解释实际成本并改善代表完整训练步的方案，而不是只提高一个利用率百分比。

长期目标是在同一个模型上联合推进 next-token 质量、数据/容量/依赖深度、低活动与局部性。
当前执行成本限制单位墙钟内可取得的训练曝光。稠密计算加遮罩仍可用于学习研究，
本请求不要求先兑现活动比例对应的吞吐，也不要求先实现真实稀疏执行。

**身份边界：此次被测的是实验侧独立直接数学实现，并非当前公共基座执行器。**
这些数据为基座提供真实工作负载和待核查机制；它们没有证明基座最新提交存在相同瓶颈或回归。
与 TL-REQ-004 的旧 eager profile 相关，但新旧实现、模型和计时范围不能混算。

## 固定来源、环境与证据

| 材料 | 身份 / 入口 |
| --- | --- |
| 数学 | `0770593838a37c790a568226930567ed60f99b83` |
| 实际训练源码 | `6e7bd49eb5e35c15445826cac3a9b6ddda75aff4` |
| 本机固定训练树 | `.local/source-v9-execution`；提交记录中的 tree SHA256 为 `a36b64f9bdfbd00fa87cac0aaca1c861b5b8d0c298abdcf763d3605393a3de98` |
| 观测脚本与已审报告 | `8995958728167eee7a4bcc2139f5e1794cbf82ab` |
| 已安装基座元数据 | `74e7bbd62341d2ccd382194a2c52ad4fea033665`，协议 `tide-downstream-contract-v1`；不是本次直接实现的执行源码 |
| 当前基座 checkout（仅用于定位） | `a8a4ee028a5d79696d49bc47242a29c4c39aee4d`；本需求未在此提交上重测 |
| 私有环境 | `.venv-positive-floor-v1`；Python3.12.13，Torch2.12.0+cpu，TorchNPU2.12.0.post3，`ascend/dev-workspace-9.2.0 r1` |
| 精度与线程 | FP32；训练配置 ATen threads=2，作业 BLAS/OMP 环境=4，声明 CPU quota=4核；这些是不同口径 |
| 固定环境锁 | 训练树 `dependencies/environment.lock.json`，SHA256 `3662d5f7befd9f1a7aa5da6e0f6d65337ad3daa05d58a1a6854f10962e8ec30b` |
| 固定安装锁 | 训练树 `dependencies/installation.lock.json`，SHA256 `aaa03d8367a04080f9852ba883dc3c03c41f8946a95d99fd017c52e96ca86b45` |
| 数据 | `runs/v9-data-r3/data-lock.json` 及同目录 `train.u16` 等；独立文档块，L256，自回归 next-token |

本机下游根目录：`/mi/data2T/zlong/home/llm/tide-learning-lab`。上述相对路径均从此目录起，
“训练树”路径从 `.local/source-v9-execution` 起。作业放在 `background.slice`，单进程使用映射后的 `npu:0`。
后继安装或诊断使用新目录，不覆盖正在运行的源码、依赖、配置、数据或 checkpoint。

固定 Git 入口：

- [中期人工审阅](https://github.com/ZichaoLong/tide/blob/8995958728167eee7a4bcc2139f5e1794cbf82ab/docs/studies/direct-v9-interim-20261112.md)
- [原始8轮设备计数器、job/config/step 映射](https://github.com/ZichaoLong/tide/blob/8995958728167eee7a4bcc2139f5e1794cbf82ab/docs/evidence/direct-v9-resources-20261112.json)
- [逐运行参数、配置hash、计时、正确性分项](https://github.com/ZichaoLong/tide/blob/8995958728167eee7a4bcc2139f5e1794cbf82ab/docs/evidence/direct-v9-interim-20261112.json)
- [只读观察脚本](https://github.com/ZichaoLong/tide/blob/8995958728167eee7a4bcc2139f5e1794cbf82ab/scripts/observe_v9_resources.py)

资源观测 JSON SHA256：`d0f0d6a01728bc7f399236ead822ed27da14a97f08b69e28be5d6e8cb09d534d`。
中期逐运行 JSON SHA256：`1551f51dbf3b4dba600b544c0e51c833f049d3572eb8e0ee5762ecd3edcfbedf`。
公开文档不包含 Lens 私有入口或凭据；本请求不依赖查看服务在线。

## 实测现象与口径

采样窗口为北京时间2026-11-12 **09:28:20–09:28:56**，程序总耗时37.67秒，8轮。
命令为 `npu-smi info -t usages -i BOARD -c CHIP`；当时核对物理设备映射为 `2*BOARD+CHIP`。
以下均是该短窗口的**单卡计数器中位数**，不是整条轨迹的时间平均值。

| 物理卡 | 实际 case | AICore | AIVector | AICPU | HBM带宽 | 总NPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | hb88-pool-s22 | 3% | 17.5% | 30% | 7% | 94.5% |
| 6 | hb88-pool-s11 | 3% | 21% | 31% | 7% | 94.5% |
| 8 | hb184-quad-s22 | 1% | 27% | 21% | 3% | 48% |
| 14 | hb184-quad-s11 | 1.5% | 26% | 14% | 4.5% | 56.5% |
| 4 | tree-r32-h2-bern-s22 | 2% | 14.5% | 33% | 11.5% | 100% |
| 10 | tree-r32-h2-bern-s11 | 2% | 15.5% | 33% | 11.5% | 96.5% |
| 2 | tree-r64-h2-q3-s11 | 2% | 18% | 33.5% | 11.5% | 92.5% |

物理1也采样了，但另有其他进程；其中 AICore 中位7.5%、最大32%，不用于单任务归因。
其余卡也只拥有设备级读数，不具有进程级 kernel 归因资格。原始记录保留各轮与 min/median/max。

- 不同百分比有各自分母，不能相加，也不能充当 kernel 耗时比例或理论 FLOP/s 利用率。
- AICPU 是设备侧处理单元，不能直接解释成 host CPU fallback。
- AIVector、归约和带宽工作可能是必要计算；不能只最大化 AICore 读数来定义性能成功。
- 持续存活的服务所属 PID 在采样期间约占1.6–1.9个 host CPU 核；短进程和共享 cgroup-v1
  控制器限制了归因，尚无 host 限流或主机瓶颈的证明。
- pooled 无该 quad/bern 迭代求根路径却同样 AICore 低，因此不能先把全部成本归给求解器。

另有已经完成的64M训练轨迹（每seed为67,108,864个有效目标），seed11/22合并有效目标数除以合并更新时间：

| case前缀（各有s11/s22） | 参数量 | 物理microbatch / global | 累积次数 | 更新吞吐 |
| --- | ---: | ---: | ---: | ---: |
| hb184-pool | 51.438M | 8 / 64 | 8 | 2,057 token/s |
| hb40-wide256-pool | 49.885M | 32 / 64 | 2 | 14,851 token/s |
| tf-hb184 | 52.540M | 64 / 64 | 1 | 94,102 token/s |

计时来自固定 `tide_lab/campaign_runtime.py:83` 的 `logical_update`：包含模型/任务头前向、loss、
反向、当前模型观测回读、非有限检查、clip、AdamW及终端同步；不含外部 batch 准备/上传、
模型构造、评价、checkpoint、启动和独立日志写入。token分子是非ignore目标，不是节点Full行数。
因此表中是**同步训练更新吞吐**，不是包含所有准备工作的完整作业吞吐。

这些运行与上述计数器采样不是同一时间窗口，HB184 pooled 与采样的 HB184 quad 也不是同一配置。
不能用计数器把此吞吐差额分摊到某类算子。浅宽HB约为深HB的7.2倍，是不同模型的实际成本差异；
参数、输入曝光相近不意味着数学工作、物理batch、FLOPs或共享负载相同，更不等于等价实现加速。
尚无同窗口 Transformer 的硬件计数器或本批算子 trace。

## 可直接定位的实际配置

配置在 `runs/v9-campaign/main/configs/<case>.json`；实际运行在 `runs/v9-main-<case>/`。
两者完整hash和结果路径已记录在固定证据中。所有代表为KV/qry、L256、global64、FP32；
当前物理Full是稠密计算加遮罩，逻辑接收、逻辑Full、物理工作另计。

| case（seed11示例） | 主体 / 区域 | 宽度 / FFN | microbatch | 分配与边界 |
| --- | --- | --- | ---: | --- |
| hb88-pool-s11 | 88节点 / 22区域，每区4节点 | 128 / 512 | 16 | pooled_attention；原入口 |
| hb184-quad-s11 | 184节点 / 46区域，每区4节点 | 128 / 512 | 8 | regional_total/quad，context控制，k=2，温度1；原入口 |
| tree-r32-h2-bern-s11 | 320主体节点 / 10主体区域，另21边界节点/6区域 | 64 / 256 | 8 | 主体regional_total/bern，fixed控制，k=4，温度0.25；入口树 |
| tree-r64-h2-q3-s11 | 640主体节点 / 10主体区域，另21边界节点/6区域 | 64 / 256 | 4 | 主体regional_total/quad，fixed控制，k32→2、温度1→0.5退火；入口树 |

入口树边界各区域的配置与主体不同，展开计划是权威；`k`是强度预算，不是硬Top-k个数。
R64每更新包含16次microbatch前后向，global64不能证明某个设备矩阵算子具有B64。
退火中的模型必须固定采样时的进度、权重、状态和输入，不能只用配置的最终k代表当时工作量。

## 优先排查的机制（尚未证明是本批热点）

源码链接均固定到实际训练提交：

1. [聚合中的class高级索引](https://github.com/ZichaoLong/tide/blob/6e7bd49eb5e35c15445826cac3a9b6ddda75aff4/tide_lab/direct_semantic.py#L159)
   及同文件265–280行的节点 select/stack。核查真实前反向是否产生大量 Index、IndexPut、Scatter、
   梯度整块清零或重复装配，哪些映射到AICPU。旧TL-REQ-004曾有Index/IndexPutV3 trace，
   但旧kernel次数/耗时不能当成本批证据。改成另一种索引API也不自动保证更快。
2. [区域分配求根](https://github.com/ZichaoLong/tide/blob/6e7bd49eb5e35c15445826cac3a9b6ddda75aff4/tide_lab/direct_semantic_ops.py#L87)：
   28轮求根与active-set修正中的小归约、where、启动和临时张量；区分AIVector必要计算、
   提交开销与同步。考虑批量/融合时保留正支持、可行性和声明的隐式VJP；pooled作为不同函数定位参照。
3. [每microbatch的观测回读](https://github.com/ZichaoLong/tide/blob/6e7bd49eb5e35c15445826cac3a9b6ddda75aff4/tide_lab/direct_semantic.py#L288)
   与[训练更新](https://github.com/ZichaoLong/tide/blob/6e7bd49eb5e35c15445826cac3a9b6ddda75aff4/tide_lab/campaign_runtime.py#L83)：
   loss标量、有限检查、clip及逐参数AdamW。区分检查、遥测和optimizer的实际成本，
   可研究合并观测或批量optimizer，但不能通过删验收、漏参数或省略工作量记录制造收益。
4. 记录 GEMM/bmm 的真实形状、dtype、布局、每次调用行数、区域依赖和内存流量。
   结合HB184深窄与HB40浅宽的差异判断批量、布局与依赖成本；不可只由总参数量推断利用率。

## 建议基座如何接续、如何验收

先复用现有张量列、区域VJP、索引与批量性能证据，将本请求关联到相关工作，而非重新启动旧矩阵。
首轮可从 HB88 pooled、HB184 quad、R64入口树、TF512×16 四个代表开始；R32/Bern仅在需要
区分求解器时补充。建议每个代表先一个短trace，分前向/反向/optimizer，并用独立无插桩计时校验；
使用固定初态、输入、进度和optimizer，预热后恢复同一初态。小microbatch诊断不能替代原形状成本。
建议初轮上限60分钟墙钟、8GiB新增trace，实际设备/并行度按基座当时授权与资源选择；
到界交付证据和未知项，不重跑整条16M/64M训练。此处是可裁剪的后继建议，本次未提交诊断作业。

沿基座优化skill的加法/减法方法：直接公式 → 公共批量边界 → 代表组合，分离数学职责与接口增量；
计时包含相同观察/梯度职责。直接v9仅覆盖静态局部公式及实验侧入口，不认证基座一般图调度、
独立中间根、完整continuation或恢复。公共路径桥接应显式列出增加的职责，TF是不同函数对照。

本请求期望的交付是：

1. 固定源码/环境/输入下的有限因果归因：kernel类型、调用/耗时、真实矩阵形状、同步和内存；
   对无法解释的主要差距保留具体未知项，不由利用率直接断言根因。
2. 对候选方案明确采用、暂缓或不采纳。若实施，提供公共入口/迁移方法与同材料前后对照；
   报完整训练步、上述更新子区间和作业成本，吞吐按有效目标计，额外列物理工作及资源峰值。
   不预设必须超过TF或达到某个AICore百分比。
3. 执行、严格数值、离散语义、恢复、学习效果、性能分别报告。v9在观测时116条已启动训练
   均保留至少一个CPU/NPU严格有限失败，阈值仍为atol1e-6/rtol1e-5；小型机制及三步新进程恢复
   通过不认证长轨迹重放。有限超差经有界排查可继续明确探索，不改阈值或把原因未明写成舍入。
4. 结构/反向改动检查共享参数、输入梯度、None/连接零、候选更新、正支持、分配可行性、
   optimizer与声明恢复；路由/工作量改变须单列，不称为严格等价加速。下游在新安装中独立验收。

## 只读复核入口与授权

在下游目录运行下列命令仅查看已保存材料，不重启训练：

```sh
git show 8995958728167eee7a4bcc2139f5e1794cbf82ab:docs/studies/direct-v9-interim-20261112.md
sha256sum docs/evidence/direct-v9-resources-20261112.json docs/evidence/direct-v9-interim-20261112.json
python - <<'PY'
import json
from pathlib import Path
e = json.loads(Path('docs/evidence/direct-v9-resources-20261112.json').read_text())
for row in e['summary']:
    print(row['physical_id'], row['cases'], row['metrics'])
PY
```

实际worker命令、固定源码、环境、输入hash与unit见 `artifacts/jobs/v9-main-w*/job.json`；
各case的准入/恢复/训练结果见 `runs/v9-main-<case>/`。这些是只读材料，不原样重提交worker。
观察脚本支持 `--samples 8 --interval 4 --output docs/evidence/NEW_RESOURCE_OBSERVATION.json`，
新观察须用新路径；它测当时的活动任务，不能重现历史硬件读数，也不会产生kernel trace。

用户本次明确授权将issue放入基座目录，交由基座使用；实验分支原有文档提交/普通推送授权继续有效。
本次只交接文档，未变更基座执行代码、依赖、活动作业或公共环境。旧TL-REQ-004/005原文保留。
原v9截止与预算属于原训练批次，不转成基座新作业授权或额外强制CPU/NPU算时限制。
基座回复：待回复。下游验收：待交付后独立开展。
