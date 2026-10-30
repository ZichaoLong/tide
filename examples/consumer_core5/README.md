# 可复制到实验仓库的core-5起步示例

先读所采用基座提交的[协作规范](../../docs/downstream-contract.md)。复制本目录到实验仓库，
再从[模板目录](../../templates/downstream/AGENTS.fragment.md)取接续规则及请求/交付/验收模板。
复制后这些相对文档链接应改为依赖锁中仓库+固定提交的地址；运行代码不使用它们。

本例依赖已安装的`tidegraph`，不读基座源码、tests或scripts；使用公共planning、weighted、checkpoint接口。
当前起点为FC两层、区域宽2、内容宽8、baseline KV、region共享、B2/P2。可用`--recipe`换成公共recipe合法配置，
并先在小规模验收。本例的16MiB静态材料预算及1..8步上限是示例资源边界，不是基座模型能力上限。
它提供CPU PyTorch/Python-native、FP32/FP64、streaming/prefill和scalar/packed显式配置，拒绝不适用组合。
NPU/CUDA、resident和standalone使用基座对应入口及独立验收；本目录不宣称覆盖这些消费者。
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
