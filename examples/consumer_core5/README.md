# 可复制到实验仓库的core-5起步示例

先读所采用基座提交的[协作规范](../../docs/downstream-contract.md)。复制本目录到实验仓库，
再从[模板目录](../../templates/downstream/AGENTS.fragment.md)取接续规则及请求/交付/验收模板。
复制后这些相对文档链接应改为依赖锁中仓库+固定提交的地址；运行代码不使用它们。

本例依赖已安装的`tidegraph`，不读基座源码、tests或scripts；使用公共planning、weighted、checkpoint接口。
希望通过统一命令使用CPU/CUDA/NPU推理、训练、恢复及HTML结果时，从新增的
[配置驱动任务示例](configured/README.md)进入；它有独立任务包和实际验证范围。下述旧示例继续演示应用自行组合checkpoint。
当前起点为FC两层、区域宽2、内容宽8、baseline KV、region共享、B2/P2。可用`--recipe`换成公共recipe合法配置，
并先在小规模验收。本例的16MiB静态材料预算及1..8步上限是示例资源边界，不是基座模型能力上限。
它提供CPU PyTorch/Python-native、FP32/FP64、streaming/prefill和scalar/packed显式配置，拒绝不适用组合。
前述`train.py/check.py/dependency.py`只声明这个CPU起步范围；后文的密集查询、张量窗口和
resident消费者分别声明自己的设备/API与资格，不能互相继承。standalone沿对应公共C++入口验收。
[固定源码检查报告](../../docs/evidence/downstream-contract-20261030.md)给出8组实际通过配置、拒绝检查及原失败；上游交付后仍须做本实验自己的验收。

## 固定安装及锁

在基座固定checkout构建wheel，记录完整提交及源码洁净状态，保留wheel hash和构建记录。
构建写入setuptools临时文件，使用可写构建副本；实际运行依赖保持冻结。在实验私有且Torch已配套的环境中：

```sh
python -m pip install --no-deps /path/to/tidegraph-0.2.0-py3-none-any.whl
python dependency.py --wheel /path/to/tidegraph-0.2.0-py3-none-any.whl \
  --source-commit FULL_FOUNDATION_COMMIT \
  --math-commit 0770593838a37c790a568226930567ed60f99b83 \
  --output foundation.lock.json
```

依赖锁生成器逐文件核对wheel与安装目录，记录实际Torch/Python/ABI和内容hash。
`--source-commit`是构建方提供的来源声明，字节匹配不自动证明wheel来自该Git提交；保留对应构建证据。
协议文档默认来自同一提交，需要另选时用`--protocol-commit`。未提交开发wheel用`--development-dirty`并保留补丁/归档。
锁不覆盖已有文件；正式升级先生成候选新锁并检查，通过后由实验仓库原子替换其选定锁。

Python-native在生成锁时加`--native-library /matching/cpu-build`，运行时也给同一内容的库目录。
要求匹配的build-manifest及adapter SHA256；库可以迁移路径。原生源码与包源码的对应由基座构建/交付证据保证。
环境或已安装内容不符会失败。当前锁工具仅示范Python wheel与CPU adapter，其他客户端按协作契约补对应构建身份。

## 运行与验证

```sh
python train.py --device cpu --foundation-lock foundation.lock.json --output-dir runs/first
python train.py --device cpu --foundation-lock foundation.lock.json \
  --resume runs/first/checkpoint.pt --steps 1 --output-dir runs/resumed
python check.py --device cpu --foundation-lock foundation.lock.json --output-dir checks/first
```

示例默认native worker1，可用`--workers`显式更改；autograd多线程为true，另记应用执行策略身份。
FC/KV FP32的两次worker2检查在本例新进程恢复中出现逐值差异；不能从同进程恢复通过推断所有worker设置的跨进程精确恢复。
默认FP64/packed prefill；切换可用`--dtype float32`、`--client python-native --native-library BUILD`，
或`--schedule streaming --packing scalar`。每次输出目录必须新建，锁/配置及checkpoint身份/时钟预检查失败不创建最终目录。
`check.py`先比较两步独立时间主序参考的完整结果、输入/参数梯度、共享参数、SGD和scheduler，再启动三个新进程：
连续三步、两步保存、一步恢复。最后完整观察值及解码后的应用checkpoint逐值一致；不能只检查loss。
`train.py`本身只运行应用，不声称独立正确性；合格状态需要检查记录和实际配置范围。

`task.py`拥有生成数据、线性head、dropout、MSE、SGD和StepLR；这是接入示例，不是模型效果配方。
数据的torch.Generator、Python Random及dropout的全局Torch RNG均保存；head、optimizer、scheduler、
步数/位置与图权重/别名/状态/history/pending一起组成原子应用checkpoint。每窗backward后明确detach，再optimizer更新。
恢复重建owner，核对应用代码、锁、模型/任务工作负载、执行身份和图/应用时钟；仅承诺相同栈、相同配置续接。
checkpoint不保存已消费的autograd图或上一步grad缓存，下一步从zero_grad开始；恢复后下一步全部梯度仍独立比较。

本例使用小文件将图checkpoint实际字节与应用字段一次原子发布，不重算图状态。
大型任务应另设计流式/分片应用bundle，不能将本例整体CPU副本当作大模型checkpoint方案。
输出`result.json`含安装路径、依赖锁/安装内容、实际配置、模型/任务workload/执行身份和产物hash；
`observation.pt`保留最后完整执行及训练值。基座材料化产生的测试输入未使用，应用有自己的随机数据及workload_id。
实验实际使用时还须记录自己的Git提交/补丁、数据版本、运行预算及目标设备信息；本例的源码内容hash不替代实验Git记录。

## 修改与交接

修改应用后先跑`check.py`和本实验有价值的代表组合；无需遍历全部组合。
公共模块需求按统一模板提交，附这类可运行配置/复现。上游交付后在本实验检查并填写验收，再更新锁。
不要import基座benchmark脚本，或仅凭包版本/同名recipe继承资格。

## KV密集查询的安装消费者

[`dense_query.py`](dense_query.py)单独演示显式`query_execution="dense-v1"`，
支持本轮已验的CPU/NPU FP32 × Python/native。其限定域和数值边界见
[契约](../../docs/weighted-dense-query.md)与[交付证据](../../docs/evidence/dense-query-tl-req-004-20261110.md)。
前面的`train.py/check.py/dependency.py`仍只声明各自的原有范围。
复制本消费者时同时复制本目录`recording.py`；它不导入基座tests/scripts。

```sh
python dense_query.py --device cpu --client python --output-dir checks/dense-cpu
python dense_query.py --device npu:0 --client native --native-library MATCHING_NPU_BUILD \
  --output-dir checks/dense-npu-native
```

先按交付报告安装固定wheel/匹配native库并生成、验证`tidegraph lock`；NPU在本站队列内运行。
消费者对比独立CPU scalar的完整三步结果/梯度/AdamW，另起whole3、prefix2、resume1三个进程，
比较最后完整观察与optimizer逐值相等。每更新从空图状态开始，因此这里的新进程检查是
更新边界恢复；非空pending图状态/保留cut另由仓库完整图测试覆盖，不能混作同一项。
`result.json`的`execution`、`strict_numerical`、`recovery`分开审阅；有限数值超差可以记录为failed，
不能仅凭退出码宣布严格通过。该消费者不验证原FC16任务或学习效果。


批量张量窗口消费者为`tensor_window.py`。FP16显式选择
`--dtype float16 --scoring-dtype float32 --loss-scale 16`，使用FP32-master AdamW；
参数、raw状态与消息保持half，评分及loss为FP32。示例记录请求/实际路径、独立三步检查及
新进程精确恢复，半精度派生字段的比较策略同时进入身份。该示例不自动认证其他图/配置/设备。


## Resident张量块的安装消费者

`kv-bank-views-v1`中old/proposal字段默认只声明shape/present/valid；需要数值时调用
`session.observe(window)`。示例使用独立Python trace核验实际raw解码，并把proposal
keys/values作为额外独立根参与三步AdamW及新进程恢复。原始字段没有数值时也可按shape构造根。

[`block_resident.py`](block_resident.py)消费安装后的`WeightedResidentTraining`公共入口，
传入真实ports `SettleGraph`证书，并断言实际返回`WeightedBlockResidentTraining`。
当前适用域、完整字段根、只读寿命和schema-2 archive见[公共块owner](../../docs/resident-block-session.md#public-model-and-python-entry)。
同名训练入口传普通Graph时仍使用[通用记录owner](../../docs/resident-weighted-training.md)；
两者的advance输入、字段布局、step返回和checkpoint格式不同，消费者应显式迁移。

先从同一固定提交安装wheel、匹配核心库和weighted resident插件；按[开发入口](../../docs/development.md)
建立实际后端构建。以下是两个独立进程，使用新的checkpoint路径；长资格按仓库执行契约运行：

```sh
python block_resident.py --device npu:0 --native-library MATCHING_NPU_CORE \
  --resident-library MATCHING_RESIDENT_BUILD --checkpoint-path NEW_CHECKPOINT
python block_resident.py --device npu:0 --native-library MATCHING_NPU_CORE \
  --resident-library MATCHING_RESIDENT_BUILD --checkpoint-path NEW_CHECKPOINT --restore-check
```

消费者用独立Python时间主序参考检查端口presence、(c,G)、输入/共享参数梯度、
三步AdamW和实际raw Continuation；第二进程载入第一步后切面并精确比较最终checkpoint。
本小例含零质量、缺失端口、非零来源低秩因子及位置配置。它不是全部中间字段独立根、
任意模块/设备/规模或FC16性能的证明；这些由[组件与公共资格](../../docs/resident-block-session.md#navigation-and-checks)
分别验收。脚本存在和命令可构造不等于设备通过，实际固定结果从[证据索引](../../docs/evidence/README.md)查阅。
