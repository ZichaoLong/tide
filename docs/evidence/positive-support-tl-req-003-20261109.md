# TL-REQ-003 交付：Shannon 正支持数值模式

2026-11-09。接受并完成用户最新对齐的最小模式：保持 cat/ind、Shannon 和原任务配方，
显式选择最小次正规数存储，并处理 Aggregate、VJP、KV 与恢复。
原始宽范围请求见[固定下游提交](https://github.com/ZichaoLong/tide/blob/43bc5c457ed948ad81c4fbd9e41ae61516300e16/docs/requests/TL-REQ-003.md)；
任意尾部比例的 log/缩放表示、全后端实现不包含于本次交付。
[机器可读证据](positive-support-tl-req-003-20261109.json)保留失败与终态，不依赖原实验目录。

## 来源与安装

| 项目 | 身份 |
| --- | --- |
| 固定数学 | `0770593838a37c790a568226930567ed60f99b83`，19份快照校验通过 |
| 实现 / wheel 构建 | `2c11695ca66a5dcd028f7d777ec26b177add0378` |
| 零接收补测 / 通用CPU资格探测 | `2502fa4a3df3096e3f5773a7c8586b517b96da89` |
| 资格脚本筛选修复 | `b4879ffa6bceb989f052484f18cfb480f7d1a68f` |
| 包源码关系 | 三提交的 `python/`、`pyproject.toml` 完全相同；后两项复用同一 wheel，未声称重新构建 |
| wheel | tidegraph 0.3.0，436068 bytes，SHA256 `3864acb62703b4b8e58d0e7946270f9816fb2f0f5d222f915364997dd2e29ae3` |
| 实际安装文件身份 | `7dbfde4bbbd375e17e320b75a5796164cafdb33c5cfbc07de9fc02d020cd059a` |
| 安装锁 | `5e663710d6515931c4510c24a14277e0b84aeec9f17d6a8d21d078d51419e599`，pip check / lock verify通过 |
| 实际环境 | aarch64、Python3.12.13、Torch2.12.0+cpu、TorchNPU2.12.0.post3、CANN9.2.0 |
| NPU | Ascend910 / 9392；队列 physical1 → logical npu:0；单设备，无跨卡通信 |

只在新私有资格环境安装；未覆盖实验仓库的依赖、源码、checkpoint或活动作业。
本次没有 C++ 构建、native/standalone/resident 新模式执行或 CUDA 资格。

## 实现和数值边界

公共配置 `support_policy="positive-floor-v1"`，默认仍是 `strict`。
[完整契约](../positive-support-policy.md)记录前向、声明 VJP、适用域、身份与失败行为。

实际 HB16 行：
`[4.6102976799,-0.1144850254,-21.5642910004,14.2248239517]`，
caps全1、B=1.9164364337921143、tau=.25。
CPU/NPU新模式均得到
`[0.9164376855,5.6795763648e-9,1.4012984643e-45,1]`。
第三项独立80位Decimal解约 `3.1058796574e-46`，原FP32模式仍严格拒绝；
总量残差阈值未改。原始行、来源hash在[小型fixture](../../tests/data/tl_req_003_shannon_row.json)。

新模式只把数学正值舍入成零的坐标补至dtype最小次正规数；相对补入质量/cap守卫另行拒绝
明显失真的弱总量。Aggregate先归一化；Shannon VJP使用**未补值**响应；
KV qry/Full融合mass-weighted Attention的反向，避免中间上溢/下溢。
非有限与不可表示公开伴随仍失败，不作为探索超差放行。

多个尾部补为同值会失去原比例。后继运算按已存质量求值；
本模式不承诺任意实数尺度或完整网络梯度与无限精度程序等价。
例如两个最小质量、内容0/1的平均值可以前向得到.5，但质量伴随超FP32范围时明确拒绝。
这是保留的实现域，不以只看loss/最终输出隐藏。

## 实际验收

| 范围 | 结果与含义 |
| --- | --- |
| CPU新策略定向 | 29项 + 零接收补测1项通过，零跳过 |
| CPU相关默认路径回归 | 314项通过：分配/熵、Aggregate、KV、完整调度、query/共享、placement、planning/配置和拒绝快照；保留原FP64/FP32阈值 |
| NPU流程 | 29项 + 零接收1项通过；其中**13项实际以NPU为目标**，17项为CPU公式/静态配置辅助，不将30项全部计为设备测试 |
| 独立局部数学 | 实际行80位Decimal、未补值VJP的大伴随根、FP64范围反例、普通尺度gradcheck与原映射比较通过 |
| 完整小图 | 两区域/6节点/5物理边（含重复边）、B2不齐时间输入、宽2 KV sum/qry/位置；scalar独立调度与packed候选的完整记录/状态/history/pending/控制/消息通过 |
| 独立数值参考 | 从同一FP32权重转CPU FP64，独立scalar运行；完整前向/梯度按原FP32容差比较通过，没有参考轨迹喂给候选 |
| 声明训练 | 两步AdamW、全部参数梯度、共享base及私有适配、独立None/零根、保留图跨窗通过；是机制检查，不是下游学习训练 |
| 恢复 | CPU/NPU各两个新进程保存/加载：raw状态/参数/optimizer槽及下一完整窗、梯度、下一optimizer更新严格 `atol=rtol=0` 通过 |
| 真实零/缺失 | 零payload是存在的事件，但不追加KV、不激活、不发送；缺失输入没有事件，CPU/NPU均通过 |
| 已安装公共API | CPU/NPU实际行消费者执行、完整梯度/补值计数输出通过；不从源码editable安装借路径 |
| 默认兼容 | 四个历史Regional/Describe身份直接对比85d161d保持；新策略的graph/model身份不同，strict checkpoint直接跨模式加载仍拒绝 |
| 原HB16训练、FC4/FC16历史失败、学习效果、性能 | **未运行/未重新认证**；不由小图资格推断原训练已恢复或NLL改善 |

原图编码与checkpoint v5不变。仍使用磁盘完整诊断返回，不改输出范围、checkpoint周期、
optimizer边界、int64时间/边身份或固定数学。新模式的公共数值异常有原行诊断；正常计数只在
`observe_support().metrics()`显式取回，并标记包含autograd重放。

## 失败和任务生命周期

| 服务 | 固定源码 | 终态 |
| --- | --- | --- |
| gef-tl-req-003-cpu-a | 2c11695 | **failed/exit1**：新策略29通过；回归314通过、8个未加载native库的测试失败 |
| gef-tl-req-003-npu-a | 2c11695 | passed/exit0：定向29、安装消费者与上游校验完成 |
| gef-tl-req-003-boundary-b | 2502fa4 | passed/exit0：CPU/NPU零接收各1 |
| gef-tl-req-003-cpu-c | b4879ff | passed/exit0：正确筛选的8个Python失败快照检查、CPU安装消费者、上游校验 |

CPU-a的8项不是数学超差：旧测试的参数ID是True/False，`-k 'not native'`漏选了True分支，
报 `ModuleNotFoundError: _tide_native`。b4879ff把该文件单列并明确筛选Python分支；
只补跑受影响的8项，未重复314项已通过范围。原CPU-a外层任务仍为failed，不重标通过。
当前资格脚本全流程在修正后未整体重跑；以上列出各固定分段实际证据。

开发期另保留：首次误选native的87通过/3加载失败；一次60秒交互上限终止，均不算完整资格。
测试中Decimal转换requires_grad张量的告警来自只读独立参考；TorchNPU TypedStorage告警来自
checkpoint适配的弃用接口，两者均未隐藏异常或更改数值断言。

四项正式服务均使用只读固定工作树、独立installed wheel、background.slice、KillMode=control-group、
CPUQuota1核、ATen/BLAS1线程、MemoryMax8GiB；主流程各1800秒，补测900/600秒。
NPU队列单卡等待上限600秒，两次实际均约6.65秒入场；不停止其他项目进程。
最终全部MainPID=0、ControlGroup为空且相应cgroup不存在；原失败单元保留。
无性能计时；测试耗时13.56/36.78秒等仅为校验过程成本。

## 下游接入和复验

这是基座交付，**tide-learning-lab尚未验收**。原需求仍在实验仓库；不代其更新锁或重启训练。
在干净检出b4879ff或包含它的交付提交中，选择新的匹配私有环境：

```sh
# 本机先加载自己选择的匹配公共基线；TIDE_VENV/TIDE_WHEELS由调用者设置为新路径。
: "${TIDE_VENV:?设置新的私有环境路径}"
: "${TIDE_WHEELS:?设置新的wheel输出目录}"
python -m venv --system-site-packages "$TIDE_VENV"
source "$TIDE_VENV/bin/activate"
python -m pip wheel --no-deps --no-build-isolation . -w "$TIDE_WHEELS"
python -m pip install --no-deps "$TIDE_WHEELS/tidegraph-0.3.0-py3-none-any.whl"
tidegraph lock create --wheel "$TIDE_WHEELS/tidegraph-0.3.0-py3-none-any.whl" --output installation.lock.json
tidegraph lock verify installation.lock.json
python examples/consumer_core5/positive_support.py --row tests/data/tl_req_003_shannon_row.json
```

重新构建的wheel记录自己的hash；实际Python包身份、来源与后端检查分别核对，不能只看版本号。
基座定向验收入口为 `bash scripts/qualify_positive_support.sh NEW_OUTPUT cpu|npu`：
在干净固定工作树、已安装wheel中运行；本机长任务用background.slice，NPU由站点队列分配。
它没有站点绝对路径，不依赖本次artifacts。若做开发检查，可直接运行契约列出的pytest路径。

应用接入仅增加对应 `RegionalConfig` / `model.recipe.region_overrides` 的策略字段，
保留cat/ind、原数据/head/loss/AdamW配方，记录新model_id/graph身份及补值计数。
先做自己的小型任务check，再做原材料的有界延续，分别报告执行、数值/离散、
精确恢复、NLL/实际Full/消息和成本；不要求每步跑CPU参考。

旧strict checkpoint不能直接按新身份读取。先用原固定安装解码和保全；若要迁入权重/训练材料，
由下游显式定义数值模式转换并创建新运行记录，不直接改identity冒充原轨迹精确恢复。
更完整的跨模式迁移或多尾部比例表示有需求时再设计；同新模式自身恢复已另行通过。
