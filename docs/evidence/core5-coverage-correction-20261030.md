# core-5 实际配置覆盖与证据导航修正

数学固定 `0770593838a37c790a568226930567ed60f99b83`；实现固定
`0f4506d097cbae88d4cd7f1c9d31319a2e0a1f4a`。本轮修正测试覆盖、配置说明与查询，不改数学函数、
图/checkpoint版本、数值容差或执行算法，也不新增性能实验。
[机器证据](core5-coverage-correction-20261030.json)从实际JUnit提取52项检查及26个完整配置，
另存环境、源码/二进制/日志hash和开发失败；[配置证据索引](core5-index.json)连接本报告与已有性能记录。

## 已验证范围

只读固定提交，aarch64、Python3.12.13、Torch2.12.0+cpu：**73 passed，0 failed，0 skipped**，119.591s。
其中52项为下表26种recipe分别用纯PyTorch与Python→native；其余21项验证planning、目录、配置事实及证据查询。
这是相关检查的资格，未重跑仓库全矩阵；测试耗时不作性能数据。

| 实际模型组 | recipe数 | 主体节点/区域/边 | 验证的差异 |
| --- | --- | --- | --- |
| FC两层 | 7 | 4/2/4 | 五种单记忆；另有KV bern、ent |
| FC四层 | 1 | 8/4/12 | KV textbook：位置与窗口配置 |
| FC八层 | 2 | 16/8/28 | baseline/textbook混合，实际包含KV、GDN、SSD |
| HB | 9 | 8/4/16 | 五记忆/混合、textbook KV/混合、regional_total_zero |
| LH启发 | 7 | 8/4/18 | 五记忆/混合、unit_attention |

共同设置为内容宽8、FFN宽32、区域宽2、B2/P2、seed7、CPU FP64载荷及评分、packed prefill、ATen算子。
ATen/BLAS线程1，纯PyTorch worker1，native workers2。不同模块的全部key/value/head/conv/窗口/共享参数
见机器记录的`models`分组，实际可训练参数为8,190–33,356；不能因公共宽度相同而认为各模块参数量相同。

每项都断言层间边、可达性、同秩区域（HB/LH）、生成与材料化配置一致、实际模块族及执行事件。
FC混合保留rank4/8两个中间区域；textbook另外断言位置频率以及适用的qry/att设置。
随后连续两步SGD（lr=.001、momentum=.3、weight_decay=.01，每窗detach）对照独立时间主序参考：
完整返回/状态/history/pending/控制/消息/端口、loss、全部参数、梯度含None/零及SGD槽。
原owner保存到磁盘，新owner加载后核对精确状态和第三窗完整返回/梯度/optimizer，atol=rtol=0。
跨调度浮点仍使用原FP64 atol=1e-10/rtol=1e-8，整数和离散身份严格；没有放宽容差。
独立调度检查不代替各模块的手算、独立公式与梯度根测试，后者从[semantic-map](../semantic-map.md)进入。

## 纠正的历史表述

- 原24项recipe测试的FC全部depth=1，无主体层间边；FC mixed实际只有KV。原通过数保留，
  [旧报告](core5-study-clients-20261029.md)附纠正说明。本次多层/异构资格仅属于本次固定源码。
- [自然规模报告](core5-natural-study-20261029.md)的KV窗口并非统一2048：两个FC medium为2048，
  HB medium为0/2560，FC 1B为0/5120，HB/LH 1B为0/4608。0表示不裁剪，非中心窗口为512×逻辑位置步长。
  现在表格从所有节点配置生成，并由独立预期值与生成表回归核对。
- SrcClass默认按来源节点（内部source+1，外部0）分类；`SourceBinding=region`另外控制A/B按目标区域共享，
  embedding按(region,class)绑定。参数分类、参数共享和节点私有状态保持分开。

原25个性能配置仍只有7项完整通过，checkpoint后续1项仍失败。没有重写历史JSON、测试计数或测量结果。
本次没有新增standalone LibTorch、FP32/FP16、mixed/resident、NPU/CUDA或大规模资格；原数值、异步及资源边界保留。

## 查询及维护

在Python≥3.10环境中，以下命令不导入Torch，也不读取旧artifacts：

```sh
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --topology fc --family ssd --kind correctness --format tsv
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --kind performance --device npu:0 --format tsv
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view models --case fc-1b-kv --format json
```

索引有52个正确性记录和26个性能记录（25原测量+1checkpoint后续），不把这些记录加总为一个门禁。
源码/model_id/workload_id/execution_id保持独立，完整值在JSON中；测试记录永不产生吞吐资格。
失败、跳过、缺少训练/恢复检查都不能由名称或耗时提升为通过。
尚无逐配置机器记录的历史资格仍从[证据总目录](README.md)和模块契约进入，不伪造拆分；未出现的组合只是此索引未认证。
已在不含Git/build/artifacts的源码导出中用`python -S`和空PYTHONPATH验证以上查询：52项正确性、26项性能、7项合格性能和19项失败性能均保持；详情在机器记录的`navigation_validation`。
新run自动保存完整配置事实；新增recipe检查应断言实际结构和执行模块，在检查通过后才写`core5_checks`。
维护表见[core5-study](../core5-study.md)，经验见[performance-lessons](../performance-lessons.md)。

## 固定来源与新checkout复现

没有C++改动，复用b255 CPU构建前后均核验C++源hash和manifest内全部二进制hash；构建不是本次新构建。
C++源hash为`d6640489b1351a9a4970f19bd57eed78a851afa8bc60cbc488d785126f8ffb42`，
native模块为`95fbda055690bc78e43e64f4d4adf2a536f3a11dabd6ad8df754875a602251c8`。
开发与固定资格在独立background.slice、Nice10、CPUQuota3200%、128GiB、1200s服务执行；没有并行性能计时。
服务提交时核对独立PID/ControlGroup；终态服务已回收，以持久job退出码、日志、JUnit和源码守卫共同判定，不凭失效unit默认字段判通过。

开发首轮52失败/21通过来自新增测试变量重名，修正后73通过；其原源码归档及失败日志hash保留。
固定资格任务`core5-coverage-qualified`的源码洁净，归档内容未变化；机器记录保留两个开发任务及资格任务身份。
这些本地目录不是新checkout运行依赖。用匹配Python/Torch环境检出上述实现提交，按[development](../development.md)建立环境后：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --backend cpu --build-dir build/cpu --jobs all
TIDE_BUILD_DIR="$PWD/build/cpu" PYTHONPATH="$PWD/python:$PWD/build/cpu" \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q --dtype both tests/test_core5_recipe_training.py tests/test_core5_coverage.py \
  tests/test_core5_study.py tests/test_core5_planning.py \
  --junitxml=artifacts/coverage.xml -o junit_family=legacy
python scripts/record_core5_checks.py --junit artifacts/coverage.xml \
  --source 0f4506d097cbae88d4cd7f1c9d31319a2e0a1f4a --output artifacts/coverage-checks.json
```

以上是workload命令；长构建/检查仍使用固定源码、独立服务和资源预算。新环境结果须另留身份，不继承本机资格。
