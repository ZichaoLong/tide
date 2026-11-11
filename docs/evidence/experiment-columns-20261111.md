# 公共实验张量窗口与有限代表成本

实现固定 `b993f38f264f9703f0656bb1502f2f85751f8c7f`，数学固定0770593。
原始资格/成本摘要及配置、身份、计数和哈希见[机器记录](experiment-columns-20261111.json)，
实际源码审阅见[7文件/56函数manifest](experiment-columns-20261111-reviews.json)。
自动清单的2139文件/5002提取函数不等于已审覆盖，C++未配置解析。

## 交付与正确性

内置输出任务的公共CLI现在直接构造来源共享列，经物理端口gather进入B×L窗口；
loss逐物理输出端口计算`sum((cG)^2)*scale`，反向之后才整列复制、无梯度解码CPU输出。
`input_interface=auto|records`属于execution-v5；旧冻结记录缺失此字段时保留record身份。
插件、完整事件观察及能力后备不被静默改成另一种根。相关契约见[张量窗口](../weighted-tensor-windows.md)。

固定资格133 passed。8个安装消费者（CPU FP64/NPU FP32×KV/SSD×PyTorch/python-native）
各完成独立标量三步训练及CLI三个新进程first/resumed/whole；恢复全部passed。
原harness的strict numerical为failed，原因是把返回的Python float loss作精确比较。
逐项审核确认所有差异均为`window[i].return[1]`，Tensor、离散字段、状态、梯度和optimizer无其他差异；
按原dtype和既定容差离线复核loss全部通过。原failed保留，新增结论单列，未重跑或改阈值。
开发dev1/dev2测试配置错误也保留，不计作语义失败或通过。

这与下述NPU精确下一窗失败不同：后者确有参数更新逐位尾差，不以容差修正。

## 默认CLI的代表训练成本

FP32，B4/L16、staggered-v1，实际61输入位置；一次预热、一次测量，SGD。
默认python-native、packed算子策略和finite检查、tensor-window-v1输入/loss、outputs返回。
CPU用1个ATen线程，NPU用2个，CANN9.2/Torch2.12/NPU物理11、逻辑npu:0、mixed-c。
计时包含输入准备/上传、调度、packing、前向/loss、反向、检查、更新、结果返回和同步；
构造、磁盘checkpoint、独立验证和恢复检查在外。成本不是严格大规模等价吞吐。

| 代表 | 节点/宽度 | CPU秒（位置/s） | NPU秒（位置/s） | 原NPU精确下一窗 |
| --- | --- | ---: | ---: | --- |
| 宽FC/KV | 4/512 | 0.2786（218.9） | 0.2373（257.0） | passed |
| HB/KV | 176/32 | 2.7309（22.3） | 8.8803（6.9） | passed |
| LH启发/KV | 176/64 | 3.4760（17.5） | 9.0272（6.8） | passed |
| FC/KV sum | 16/64 | 0.2422（251.8） | 0.3960（154.0） | passed |
| 四层SSD | 4/128 | 0.1268（481.1） | 0.2127（286.7） | failed |
| 四层GDN | 4/128 | 0.2263（269.5） | 0.2300（265.3） | failed |
| LH启发/五记忆混合 | 176/64 | 4.6581（13.1） | 10.0518（6.1） | failed |
| LH启发/KV，PyTorch | 176/64 | 4.7126（12.9） | 10.7112（5.7） | passed |

CPU全部精确下一窗passed。CPU32线程宽FC为0.1489秒，多节点LH为3.9653秒：线程增加并非对所有形状有益。
默认这些KV代表使用sum；不能借用主FC16 qry的身份。完整记忆、区域配置与独立参数量见JSON，
不将输入位置、节点更新、Full和消息数量混作吞吐分子。
例如CPU HB候选/Full为6340/3466，LH为8416/2727；NPU LH为8413/2724，跨设备真实轨迹不同。

CPU/NPU任务部分同时运行；虽然服务请求不同CPU池，子进程记录实际可见320CPU，不能称亲和隔离已生效。
实际ATen线程如上，外部争用未排除。后续正式计时用进程实际affinity复核资源约束。

NPU SSD失败字段`nodes.0.kernel.projection`为21/49280元素，max7.45e-9；
GDN同字段4/32768，max2.91e-11；混合LH的`nodes.0.kernel.base`为5/8192，max3.73e-9。
保留已完成的成本和failed精确恢复，不能把外层执行completed读作全部通过。
有界probe中普通重复索引VJP不逐位确定，诊断用deterministic开关后SSD/GDN双新owner相等。
局部修复与其新源码设备资格另列，不能回填本报告为原实现通过。

## 四层独立公式对照

B4/L32/d128、4层、FP32、proportional单位发送；实际128输入位置。
同材料、同任务投影；tensor基线省略辅助Describe、一般区域/图观察，不能作完整函数等价oracle，
也不是标准Transformer/Mamba。三模式分别为无梯度推理、含grad前向、完整训练，单位秒。

| 设备/记忆 | 默认native窗口：推理/含grad/训练 | 独立tensor公式：推理/含grad/训练 |
| --- | --- | --- |
| CPU1 KV | .02209 / .02351 / .04365 | .01030 / .01186 / .02820 |
| NPU2 KV | .03663 / .04777 / .04915 | .01180 / .01236 / .02378 |
| CPU1 SSD | .04382 / .04912 / .10366 | .01505 / .01722 / .03864 |
| NPU2 SSD | .06005 / .05765 / .08480 | .01418 / .01591 / .02898 |

这24点是有限成本，对照仍显示图/辅助计算与接口增量；不凭总差额给某一函数分配因果占时。
SSD训练原NPU代表的恢复失败独立保留。NPU主矩阵因恢复失败未进入层链，followup只补了未运行的12点，
没有重复已通过成本。主FC4/FC16与Transformer、warm TTFT仍见[前述报告](tensor-windows-20261111.md)。

## 约1B/9B规模推理

相同b993f38安装，LH启发176节点，KV sum与教材区域配置；FP32、SGD不启用、无梯度推理，
一次预热一次测量，输出投影。CPU64与NPU2顺序运行，taskset实际亲和80–143，共64CPU。
NPU空闲物理11；项目内无重构建/其他正式计时重叠，外部争用仍未排除。

| 独立参数量 | B/L与实际位置 | CPU秒 / 位置每秒 | NPU秒 / 位置每秒 |
| --- | --- | ---: | ---: |
| 1,051,337,338 | 32/16，481 | 20.5421 / 23.42 | 9.7257 / 49.46 |
| 9,314,607,738 | 4/16，61 | 8.0040 / 7.62 | 3.8944 / 15.66 |

9B候选/Full/消息两设备均6738/1403/8395；1B CPU为55029/12204/73469，
NPU为55029/12205/73474，仍有轨迹差异。NPU测量峰值allocated约6.38GB/40.55GB。
较小B/L的9B不能与1B单步耗时直接作规模效率比；它们也与历史LH add/Attention数学不同。
四点全部完成，但独立规模数值与精确下一窗均未检查，不能称合格等价吞吐或训练能力。
构造、最终checkpoint另计；大文件写入显式tmpfs，结果archive持久化，tmpfs不认证重启恢复。

复现使用同一生成器case=lh-1b/lh-9b，samples=32/4、positions=16、mode=inference；
CPU threads=64/NPU threads=2，finite_check_packing=packed，checkpoint_interval=2、checkpoint_check=false。
resources.storage_medium=tmpfs，单点720秒，总作业3600秒/512GiB，依次执行四点。完整配置摘要见JSON。

## 重现与维护

干净检出此固定提交，按[开发入口](../development.md)安装匹配Torch/LibTorch，再运行：

```sh
TIDE_BUILD_DIR=BUILD python -m pytest -q tests/test_experiment_columns.py tests/test_selected_boundary_execution.py tests/test_selected_boundary_capacity.py tests/test_component_window_policy.py tests/test_experiment_engine.py tests/test_experiment_config.py tests/test_experiment_finite.py tests/test_task_observations.py tests/test_experiment_task_module.py tests/test_experiment_placement.py tests/test_prefill_regression_config.py --dtype float64
python scripts/prefill_regression_config.py --suite tensor-windows --device npu:0 --threads 2 --output NEW_MATRIX.json
# 显式site绑定matching native_library、安装环境；将finite_check_packing设为packed后用公共matrix入口执行。
python -m tidegraph matrix NEW_MATRIX.json --site SITE.json --output-dir NEW_RUN
python scripts/benchmark_layer_chain.py --family kv --implementation native --interface window --device npu:0 --threads 2 --mode training --batch 4 --length 32 --width 128 --native-library BUILD --output-dir NEW_CHAIN
```

用同一生成器选择SSD、CPU1、tensor及其三模式；无需遍历配置组合。
完整配置和实际身份保存在机器记录，临时构建/日志/checkpoint不作为重现依赖。
公共输入迁移维护`column_task`、`eager`、`materialize`、execution身份与对应独立检查；
当前待办及各放置/规模的限制仅见ROADMAP。
