当前补齐作业已启动；新ID、CPU选型、Lens、停止与恢复见[补齐操作页](language-completion-v5-running.md)。以下是旧v5第一次终态及原提交记录，不代表当前没有活动作业。

---

以下为历史快照；当前状态以上方为准。

# v5当前操作状态：全部终止

2026-11-09 14:36核对：无运行、排队或自动后继；HB16 sparse12:36:54、B FC8 13:43:33结束，所有cgroup/已知后代清空。预算监督均completed，最终22931.15秒=6.3698/24计算worker小时；原截止22:49:13不变。终态结果见[总报告](language-depth-v5-results.md)与[证据](../evidence/language-depth-terminal.json)。

Lens主要新增完成曲线：

- `tl-lm-a-hb16-sparse-v5-train-tide-sparse-s11`：A HB16 quad128，TL-A-TOPO。
- `tl-lm-b-fc8-v5-budget2-train-tide-dense-s11` / `...-train-tide-sparse-s11`：B FC8配对128，TL-B-DEPTH。
- `tl-lm-a-transformer-cpu-v5-train-transformer-s11`：近参数Transformer2048及128前缀，TL-A-CAPACITY。

下面表格是原提交分类；其中`lm-a-controls-v5`原链没有执行Transformer训练，应查看上面的独立CPU运行。FC16 sparse/HB40没有学习曲线。查询状态/日志仍有效，不要重启已完成服务来查看结果。本次未新增Lens写入或更改服务策略。

---

以下保留原运行交接与操作方法。

# v5多区域实验：运行类别与操作

2026-11-09 12:15补充：`lm-hb16-underflow-v5`已完成有界定位，实际第87步失败行及分析见[正支持研究页](allocation-support-v5.md)。新监督`lm-depth-budget-diagnosis-v5`覆盖原费用加诊断；当前权威累计见`runs/lm-depth-budget-diagnosis-v5/state.json`。诊断源码9ebbc63/`.local/source-language-underflow-v5`，CPU1/8GiB/1800秒上限，实际571.69秒，后代清空。无新增Lens学习投影，原dense仍失败。FC16原链在context-reset失败，sparse未训练；旧v4已完成，条件FC8正式训练跳过。当前其他活动源码不变。

2026-11-09 11:44排程更新见[独立CPU研究表](language-depth-parallel-v5.md)：原浅宽FC4和HB16 dense分别在56/86步分配下溢失败；HB40因前驱失败未训练。原Transformer和HB16 sparse已拆为新的CPU独立作业，分别`lm-a-transformer-cpu-v5`、`lm-a-hb16-sparse-v5`，学习前缀为`tl-作业ID-`。另有`lm-b-cpu-cal-v5`短程CPU计时。新增总预算监督`lm-depth-budget-parallel-v5`包含原全部费用；活动原链不改，下表保留原提交身份。

本批已获授权并后台实施。共同截止北京时间 **2026-11-09 22:49:13**；24累计计算worker小时，CPU训练替代相应NPU时间计入同一上限。最多6个同时计算的worker（含旧v4），等待依赖的服务不分配计算；本次最大请求仍不超过24计算线程/144GiB。最新生命周期、源锁、指标和实际Lens ID见[交接快照](../evidence/language-depth-handoff.json)。

## 在Trackio Lens中查看

项目 **`tide-learning-lab`**，搜索运行名中的 **`v5`**。分类字段是运行配置的`study_id`；当前独立记录器没有填写Lens原生Group，所以按下面的运行名前缀查找最直接。完整显示名会附加`-projection-…`，实际ID以对应`projection.json`和交接快照为准。尚未进入训练的候选不会预先出现学习曲线。

| 类别 / study_id | 学习运行名（投影前） | 设置与作用 |
| --- | --- | --- |
| A串联 / TL-A-DEPTH | `tl-lm-a-fc16-v5-train-tide-{dense,sparse}-s11` | FC16，4区域串联，width32；CPU1；各128更新、512步LR前缀 |
| A分支汇合 / TL-A-TOPO | `tl-lm-a-hb16-v5-train-tide-{dense,sparse}-s11` | HB16，4区域、3依赖层级；CPU1；与FC16近同参数、同曝光 |
| B局部加深 / TL-B-DEPTH | `tl-lm-b-fc8-v5-budget2-train-tide-{dense,sparse}-s11` | Qwen最后一个MLP槽内FC8，两区域；NPU；128更新，13 MSE+115 CE |
| A浅宽控制 / TL-A-CAPACITY | `tl-lm-a-controls-v5-train-tide-dense-s11` | FC4/width64，616741参数；NPU；128更新 |
| A普通LM控制 / TL-A-CAPACITY | `tl-lm-a-controls-v5-train-transformer-s11` | 2 block、width96、4头、FFN512，670592参数；2048更新，并保留128前缀；日程差别明示 |
| A递归嵌套 / TL-A-NEST | `tl-lm-a-hb40-v5-train-tide-{dense,sparse}-s11` | HB40，10区域、5依赖层级；CPU1；条件64更新，先等HB16机制和验证学习 |

全部新训练seed11、B8/L32、microbatch8、accumulation1。`dense`为Shannon发送，`sparse`为可学习quad发送；不以吞吐同比改善作为准入。`history512`仍不代表512-token实际上下文。

训练主要看`train/loss`、`validation/nll`、`activity/full_fraction`、`work/full`、`work/messages`、每32步的`region/N/candidates`与`region/N/full`、步时及NPU内存。最终`test-…`是新确认集评价，`inference-…`是完整推理/数值/因果检查，不是独立训练。

`tl-lm-a-reference-v5-test-…`和`tl-lm-b-reference-v5-test-…`只把旧固定权重放到同一新确认集评价，不能算新训练或新seed。数据小记录是`tl-lm0-depth-v5`，已verified/readback_matches。新接入、微型恢复和校准保存在项目目录，不逐个投影到Lens，避免重新堆积技术记录。

私有查看入口、SSH转发及已有登录方式在本机`.local/LENS.md`。服务、认证和访问策略没有改变。

## 实测资源选择与失败

FC16/B8/L32短校准：CPU1约22.21秒/步、CPU4约32.17秒、共享NPU约80.20秒；A深层选择CPU1并行独立研究。CPU4的B16/B32分别约45.61/77.52秒每步，约11.22/13.21 token/s；CPU1/B8约11.52 token/s。B32吞吐增加有限且会改变同曝光下更新次数，当前保留B8以匹配旧采样和128步前缀。NPU较大batch的限时失败保留，不改变已启动配置。

B与容量控制使用轻载共享NPU：两次观测AICore≤20%、空闲HBM≥16GiB，持本账号设备锁，实际映射保存在queue/worker记录。其他用户可在之后改变占用，时间记录是共享资源观察，不是独占后端比较。

`lm-b-fc8-v5`完成机制、三步精确恢复与校准后，完整配对预测10664秒超过原3300秒训练分项，因此以预算准入失败终止，正式训练0次。`lm-b-fc8-v5-budget2`重新初始化相同两个候选，复用不可变cache，给单次最多4.5小时；这不是checkpoint恢复，也没有增加候选/seed/更新数。原失败及成本保留。全批实际计算由`lm-depth-budget-v5`监督，累计临近24小时或共同截止时停止本批指定服务；单作业上限之和不能当作另获计算预算。

A FC16/HB16使用同CPU配置比较。与旧FC4及NPU控制的比较同时改变了执行后端，需保留这一边界；旧权重会在相同新文档复评，不能用旧测试集分数混算。B单槽加深不代表整个Qwen局部化。独立严格数值、离散语义、恢复、学习与性能分别审阅。

新增接入的有限严格失败已单独[审阅记录](../evidence/language-depth-checks.json)：A FC16 dense/sparse分别619/2810个超差元素，HB16为900/381，B FC8为14/144，浅宽FC4为92；所测最大绝对差不超过4.77e-5。保持core原1e-6/1e-5阈值与未知原因，不能称严格等价。完整轨迹/VJP、支持、共享输入、未来VJP、各区域参数依赖及三步精确恢复分别检查后，继续已授权FP32学习。CPU选择基于实测成本，也存在严格超差，并非为了消除数值差异。

## 固定源码与状态

| 作业 | 冻结源码 | 上限 / 等待 |
| --- | --- | --- |
| 数据、CPU/NPU短校准、A容量控制、B首次尝试 | `100b4507b0c4a9bb359e3238f5f6fe1ecb7c0f59`；`.local/source-language-depth-v5` | 数据15min；校准各30min；容量控制90min；B首次已失败终止 |
| A FC16、HB16、条件HB40 | `aa76da565a4612b8d009a7abedbbd1187ef0c95a`；`.local/source-language-depth-cpu-v5` | 每研究最多6h；HB16等待CPU校准释放并发位，HB40等待HB16完成 |
| B预算调整、旧权重复评、预算监督 | `8626ab2bcfea2f091341bf00e511c2b93d512bfb`；`.local/source-language-depth-budget-v5` | B≤4.5h；A/B复评各30min；复评等待已有校准/复评释放并发位 |

数学0770593838a37c790a568226930567ed60f99b83、foundation85d161d5c9148eac2b164f89a0ba4204b4331898及安装身份保持。新16份确认文档排除此前32份，A/B训练/验证张量完全保留，tokenizer不变。所有长期输入不可修改；工作区后续文档修改不进入冻结作业。

```sh
python scripts/job.py status artifacts/jobs/lm-a-fc16-v5
python scripts/job.py logs artifacts/jobs/lm-a-fc16-v5 --lines 50
python scripts/review_language_depth.py
# 停整个指定服务及其训练/记录器后代，并核对实际终止
python scripts/job.py stop artifacts/jobs/lm-a-fc16-v5
```

其他作业ID：`lm-a-hb16-v5`、`lm-a-hb40-v5`、`lm-a-controls-v5`、`lm-b-fc8-v5-budget2`、`lm-a-reference-v5`、`lm-b-reference-v5`。候选链与原因在`runs/ID/stage.json`；CPU依赖等待在`runs/ID-wait/state.json`，NPU等待在`artifacts/jobs/ID/queue-state.json`。预算累计见`runs/lm-depth-budget-v5/state.json`。只停止监督服务不能停止训练，停止整批须逐个停止其配置列出的作业。

恢复先停止旧服务并核对无后代，读取最后完整`checkpoint.json`。使用原冻结源码、原配置、相同data/cache/线程/设备身份，以新的作业ID和输出运行`language_stage.py train`，传原kind/depth/steps/schedule及`--resume OLD_CHECKPOINT`。具体原argv在stage wrapper的记录和job.json中；CPU/NPU或NPU物理映射改变不属于已声明的精确恢复。原三步精确检查通过不等于128步独立长轨迹恢复已验证。新恢复仍计入24小时累计预算，须登记监督清单，不能另起未计费的重试。

## 已执行清理

六个旧缓存/微型恢复checkpoint已退役，回收2.0126GiB；[路径/hash/影响清单](../evidence/artifact-retirement-v5.json)。最终科学checkpoint、全部v4/v5与本地指标/失败/日志保留。

已删除44个历史技术Lens投影，删除时项目运行数161→117；另5个候选此前已不存在或身份未确认，未按名称猜删。[退役记录](../evidence/lens-retirement-v5.json)保留实际ID、本地来源及hash，旧projection报告表示当时曾核对通过，不表示现在仍在Lens。HTTP管理读取使用已有凭据返回403，本次通过同账户安装的官方Trackio本地存储接口清理，未修改凭据或认证服务。
