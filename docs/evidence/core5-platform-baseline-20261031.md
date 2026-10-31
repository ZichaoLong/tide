# 三平台首组矩阵与大batch基线（2026-10-31）

本报告审阅102条终态记录：98项正式流程与4项独立GPU观察。43项正式流程严格通过；38项只在探索策略下接纳数值超差，17项保持严格失败。102项执行、离散语义及磁盘精确续接均通过。检查范围有限，不认证约1B大batch、多卡、resident容量或全部训练。

[可直接查询的v2完整小记录](core5-platform-baseline-20261031.json.gz)保留原配置、逐窗/warmup成本、完整严格诊断、实际安装/构建身份、工作量、峰值和profile摘要。压缩只减少存储，不改变内容；原checkpoint及大trace不是长期依赖。

## 固定身份与环境

数学固定 `0770593838a37c790a568226930567ed60f99b83`。CPU32、NPU32、GPU2项规模pilot及4项观察使用`1b3e39f4f58378272d5afe41f970c92dd0e4fed7`，安装内容hash `6af9b3981aa512982618b31b02ad69db5efd20c31438b6cb273038d877c68d29`。GPU首组32使用`972032f`，安装内容hash `f89d320022b18f78bd8637d5851fd37db674c82ab04f4c8f0a36e21ae936dc48`。

CPU core来自de999c2、standalone来自eca070f；NPU core/resident/standalone来自eca070f；GPU core/resident来自de999c2、standalone来自eca070f。各构建的完整提交、源码/二进制hash见archive的details.identity.builds；这不是在较新Python提交上重建C++的声明。三平台standalone分别通过3个CTest。

本机aarch64，Torch2.12.0+cpu/TorchNPU2.12.0.post2，Python3.12.13；Python NPU选CANN9.1，独立SDK选LibTorchNPU2.10/CANN9.0。NPU队列选择物理11→逻辑npu:0。CPU默认ATen4/interop1、调度worker1；NPU ATen1/interop1/worker1。GPU为RTX3090/sm86，Torch2.12.1+cu130，CUDA13.0，ATen1/interop1/worker1，IEEE FP32，TF32关闭。两机分别串行本组计时；重型构建不重叠。非本项目争用未排除，不由单次结果宣称普遍最优。

## 已测范围和严格状态

| 平台 | 正式流程 | 严格通过 | 数值探索接纳 | 严格失败 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 32 | 15 | 12 | 5 |
| NPU | 32 | 14 | 12 | 6 |
| CUDA | 34 | 14 | 14 | 6 |

每平台覆盖FC/HB/LH的Settle、真实改变延迟的非对齐TimedDAG、真实反馈PDG；同一FC模型的合法层级嵌入单列；一般Settle端口/物理重复边/同秩区域保留。主架构KV外含HB qry/位置/Attention及LH KV/GDN/SSD异构。三客户端有共同小型推理与一阶SGD训练，完整记录、最终参数/梯度/槽及恢复分别比较。三客户端小训练在各平台均通过本配置；不解除其他输入/独立根/AdamW或workers2历史失败。PDG性能候选为native，Python保持独立参考。

CPU严格失败为HB反馈、LH反馈/非对齐/Settle/异构；NPU及CUDA还包括FC反馈。首差分别出现在Full、content/evidence、KV keys/values、conv_values或pending值，归档保留坐标、原阈值、大小与全部诊断。失败均未被探索、外层作业或自身恢复覆盖。

## 固定1,134,054参数的batch曲线

FC4节点/2区域、d128、FFN4d、KV、P2/span8，FP32 payload/scoring，native prefill；CPU placement=cpu，设备placement=mixed-c。逻辑B是独立staggered-v1样例，真实缺失输入、消息数、活动和缓存长度不同。完整返回；局部packed、返回scalar。每配置一次独立流程、warmup1窗、连续2窗中位，另外精确恢复窗和独立参考在计时外。

**下表全部保留数值严格失败，只是相同离散工作量的探索成本。** 分母为完整窗口，包括准备、传输、调度、packing、计算、finite检查、完整返回及同步。checkpoint发布、构造、参考与恢复分别计费。

| B | CPU秒 / 样例每秒 | NPU秒 / 样例每秒 | CUDA秒 / 样例每秒 |
| ---: | ---: | ---: | ---: |
| 1 | 0.018565 / 53.86 | 0.070196 / 14.25 | 0.045424 / 22.01 |
| 8 | 0.068449 / 116.88 | 0.240972 / 33.20 | 0.194626 / 41.10 |
| 32 | 0.269107 / 118.91 | 0.791068 / 40.45 | 0.568885 / 56.25 |
| 128 | 1.036419 / 123.50 | 3.319522 / 38.56 | 2.504214 / 51.11 |
| 512 | 3.888579 / 131.67 | 18.175708 / 28.17 | 8.605355 / 59.50 |

B128每窗candidate672、Full427/430、消息396/398；B512为2688、1703/1706、1584/1586。三个平台这些离散计数相同。State最大实际样例批分别128/512，Full最大时间行161–162/645–646；后者含不同逻辑时间，不是新的独立样例数。私有矩阵BMM最大仍2，跨节点独立投影最大10，不能从逻辑batch推断所有算子都已形成大矩阵。

| 平台 | B128 RSS峰值 / 设备allocator峰值（bytes） | B512 RSS峰值 / 设备allocator峰值（bytes） |
| --- | ---: | ---: |
| cpu | 491,339,776 / 不适用 | 961,036,288 / 不适用 |
| npu | 2,172,661,760 / 54,986,752 | 2,683,310,080 / 104,686,080 |
| cuda | 1,538,908,160 / 28,514,816 | 2,009,411,584 / 73,959,936 |

峰值含构造/测量/恢复阶段；RSS为进程生命周期高水位，设备allocator不含厂商/驱动隐藏分配。完整逐阶段值在archive；不是显存容量上限。

## 同材料packing和控制放置

| 平台 | A scalar/scalar秒 | B packed/scalar秒 | C packed/packed秒 | A/B成本比 | B/C成本比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 1.658510 | 1.036419 | 1.019537 | 1.600 | 1.017 |
| npu | 5.350006 | 3.319522 | 2.989671 | 1.612 | 1.110 |
| cuda | 4.190778 | 2.504214 | 1.840903 | 1.673 | 1.360 |

这些是原严格超差配置的成本比，不称严格等价加速。三策略保持模型/输入/B128/P2/精度/放置/线程/完整返回一致。scalar只关闭声明的局部时间批策略，来源/区域内部既有优化仍存在；auto未当作第三算法。当前单次大batch返回packing在CPU几乎持平，在两种设备降低成本；不能覆盖旧NPU B8反增的证据。

同B128 mixed-a/b/c：GPU1.610346/2.320591/2.504214秒；NPU2.639668/3.296418/3.319522秒。将控制放设备未在本次材料上更快，仍有主机控制/小调用开销。CPU ATen1/4/16为0.977557/1.036419/1.028984秒，调度worker固定1；这是单次观察，不是所有线程/模型结论。

## 规模pilot和独立profile

GPU FP32 B1/P2：99,487,206参数中位0.094079秒，1,010,339,302参数0.162938秒；都完整执行/离散/精确续接通过、数值超差。1B恢复检查63.03秒、allocator峰值4,788,428,288 bytes。B1不是大数值批，约1B B128/B512当时仍在独立检查，因此不计入本报告。

B128 GPU另3项packing观察和1项profile均完整检查与恢复，数值超差；它们不计入正式吞吐。KV Read12桶/672行、Full12桶/427行，实际可见长度2/3/4、Full桶6–64行、没有单行桶。raw缓存288个owner长2、192个长4；不能混同query可见桶。packed同权重MM36次/2915行，私有矩阵BMM904次/2112行，864次仅2行；scalar为MM1920次/2688行。

profile有147,380个kernel事件，kernel union245.382ms，完整窗union4900.623ms；重叠同名range不能重复相加。显式CUDA同步API39,868次、union107.931ms（2.202%），不包含所有隐式等待；_local_scalar_dense9232次、cudaMemcpyAsync46987次。isfinite inclusive958.840ms、all331.609ms、index776.628ms、nonzero596.197ms、to546.703ms。inclusive不能相加为关键路径，也不能仅以调用减少断言端到端收益。

这支持有界诊断packing试验bd2ff2e，而非直接改默认值。默认scalar保持，新增显式packed只合并有限值诊断输入，不改精度/阈值/梯度。固定bd2ff2e的78项相关检查通过（拒绝NaN/Inf、特殊视图、独立梯度、共享参数、三窗/磁盘恢复、配置/CLI/job/观察）；日志SHA256 `2d84d0a6d4f614e60f049602428246d5857cb72f421985172e1d3c84d9e040f0`。实际三平台收益仍需新实验，六项处置继续见ROADMAP及performance-lessons。

## 无旧目录复现与查询

先按[已安装流程](../human-workflow.md)从所选固定源码安装、填写实际site并核对旧库来源。跟踪配置在configs/matrices：coverage、clients、batch、packing、modules；CPU补threads，NPU/CUDA补placement和precision；GPUpilot选scale前两项。每项有原配置/身份，不需要原运行目录。

```sh
git worktree add --detach /path/to/fixed-source 1b3e39f
# 按docs/development.md构建所选目标，在新私有环境安装wheel并核对site
tidegraph plan configs/matrices/npu-batch.json --matrix --site site.json
tidegraph matrix configs/matrices/npu-batch.json --site site.json --output-dir jobs/npu-batch-new --detach
tidegraph results docs/evidence/core5-platform-baseline-20261031.json.gz --batch 512 --format table
tidegraph results docs/evidence/core5-platform-baseline-20261031.json.gz --format html --output platform-baseline.html
```

实际旧任务均终态、无存活后代；CPU/NPU矩阵外层exit1源于内层严格失败，不能改标passed。GPU四项观察/两个pilot外层passed不解除数值超差。后续源码可运行同材料，但产生新证据，不把提交映射/旧二进制冒称重新验证。

归档SHA256：`9aa83565849e7bcdb56191e04554ec79963da426cadc2ace4136f2a990202976`；102条，约2.82MiB。
