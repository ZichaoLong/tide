# 人类安装消费者、生命周期与离线报告（2026-10-31）

本报告当前含gpu3090完整人类流程；本机CPU/NPU完整流程仍在依赖链后等待，不能由CUDA流程推断通过。
公共运行器固定de93246，独立安装的`tidegraph_consumer_core5`示例拥有任务head/loss和数据进度。
报告界面单独固定1b46dc5；它只生成/筛选旧结果，不重跑或重新认证数学与设备。
数学固定0770593838a37c790a568226930567ed60f99b83；基础公共core/resident来源de999c2、standalone来源eca070f，各自实际hash保留于归档。

## gpu3090实操

从新私有venv安装固定wheel和消费者包；使用GPU1、原Torch2.12.1+cu130/sm86，项目公共环境未修改。
执行公开模块与`tidegraph`，未导入内部benchmark：环境operator探测→安装锁create/verify→消费者公式检查→配置→无大模型分配plan→基座/任务related检查→训练/推理→新进程恢复→持久job日志/停止/恢复→CSV/HTML/归档。

| 操作 | 实际结果与范围 |
| --- | --- |
| doctor、安装锁、消费者公式 | 均exit0；证明所用环境/安装和小公式，不是全基座资格 |
| 不带插件的base related | 独立推理通过，独立参数根严格失败：aggregate transforms.0.b的1/64坐标，最大绝对差2.18302e-6，原atol1e-6/rtol1e-5不改 |
| 下游任务related | 通过；独立任务head、输入/参数根、一阶训练、optimizer与完整续接，范围限该小配置 |
| 训练first与新进程resumed | 两次独立检查、完整记录和磁盘精确下一窗均通过 |
| 推理first与新进程resumed | 两次独立检查、完整记录和磁盘精确下一窗均通过 |
| job start/status/logs/stop | 最后完整窗口checkpoint已发布后停止；cancelled/143，独立SSH检查无supervisor/workload/remaining后代 |
| job resume | 从已发布checkpoint恢复到next_window8，exit0且无后代；该操作使用metrics/validation=none，未做独立数值验收，不称严格资格 |
| CSV、HTML、archive | 三种公开命令均成功；可从已审阅gzip重新生成，不依赖原运行目录 |

26个命令中base related按严格规则exit1；停止后status也按cancelled状态返回1，均不是被外层完成覆盖的通过。
外层walkthrough于05:44:06Z结束，停止任务05:43:57Z结束、恢复任务05:44:05Z结束；只记录原身份，不据此认证所有任务/规模。
原始逐命令、base失败、任务检查和生命周期记录见同名lifecycle gzip；查询归档另保存11条实际结果（包括嵌套检查、取消中间记录），不是11项独立性能测量。
下游接入通过不解除不带插件基座独立根失败，基座与消费者验收是两个事实。

## 离线HTML的固定源码资格

1b46dc5独立冻结源码10项通过：归档完整性、CSV、原记录保存、危险文本转义、离线JS筛选与公开查询一致、缺少参数不伪造、超过2^53整数的范围边界。
新的私有安装在本机生成CPU/NPU的HTML和CSV，在gpu3090从真实GPU归档生成HTML；不触碰活跃旧安装。
Firefox无头实际加载新安装输出并触发change/input事件：筛选B128、packed/packed、ATen4后恰有一条原strict-failure；最小参数2000000得到零条，清除边界后恢复一条。完整Details与失败/恢复分列可见。
首次未创建私有Firefox profile的截图尝试超时，无产品结果；创建项目私有profile后exit0并实际阅图。无头X-server探测警告与成功截图同时记录，不修改公共浏览器环境。
页面不使用外部服务/CDN；关闭JavaScript仍可读全部记录。筛选只改变可见行，不改状态、精度、归档或性能口径。
详细固定身份、wheel和产物hash见[报告资格JSON](core5-human-report-20261031.json)。浏览器及报告生成不是新的CPU/NPU/GPU数学资格。

## 从干净安装操作

完整安装、site、依赖锁、配置与任务扩展见[人类手册](../human-workflow.md)和[消费者说明](../../examples/consumer_core5/configured/README.md)。下游拥有数据/head/loss/配方；按[协作契约](../downstream-contract.md)固定依赖、基座检查、小型接入、配置运行、提交需求、接收交付、下游复验后显式升级。

```sh
tidegraph doctor --site site.json --device cuda:0 --probe operator
python -m tidegraph_consumer_core5.configure --device cuda:0 --mode training --output training.json
tidegraph plan training.json --site site.json
tidegraph check training.json --site site.json --scope related --output-dir checks/new
tidegraph job start training.json --site site.json --output-dir jobs/new
tidegraph job status jobs/new
tidegraph job logs jobs/new --lines 80
tidegraph job stop jobs/new
tidegraph job resume jobs/new --output-dir jobs/resumed
tidegraph results docs/evidence/core5-human-workflow-20261031.json.gz --format html --output human-results.html
```

CPU/NPU替换显式device及site，不能继承CUDA资格。7×T4模板仍待真机：分别探测driver/runtime/Toolkit/Torch/LibTorch/sm75、算子/精度和互联，不复用sm86产物；当前CC>=8.0 resident协议不满足T4，需要另行设计才能改变。
