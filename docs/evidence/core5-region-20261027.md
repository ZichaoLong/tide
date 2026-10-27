# Core-5 区域实际批量与 VJP CPU 资格 — 2026-10-27

固定实现/测试 `53910174d036e0809be7893527ca1b1c204a460b`，固定上游 `0770593`。
**536 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、workload exit 0 与服务 inactive/success 已核对；[命令/环境/哈希](core5-region-20261027.json)。
[完整数学与能力对应](../weighted-region-batch.md)列出定义域、返回、解码和调度限制。

Python 与独立 C++ 按真实候选数打包 B×N 区域，实际计算 Query/context、完整归一化、
quad/bern/ent 分配、三种 Attention、proportional、零温填充及硬发送。
Python 使用排序断点/受保护 Newton，C++ 使用独立有界二分；反向都是精确解映射，
不微分求解迭代，不套 HST。稳定 int64 ID 平票、逐次扣减、正支持下溢拒绝均保留。

私有图保存参数别名、每帧/候选字段根和实际分配分支；已定义 control cotangent 模式
与分支共同隔离反向组。核对逐根/混合/零 cotangent、None/连接零、重复输入、共享参数、
原位版本及一阶边界。通用引擎默认不声明批量/history 独立能力；带权 singleton history
通过显式能力接入在线调度。Python 可在 Read 已就绪后合批独立跨时间帧；C++ 当前按因果 wave
合批独立 sample owner。没有参考轨迹预跑，单行退化合法；parallel_regions 另走旧能力边界。

局部 literal quad 解/Jacobian、稳定分数填充、Attention 零 base 右导数锚定公式；
既有独立面枚举与 Decimal 解锚定所复用的标量参考。十二种正式组合在双样本 HB/反馈 PDG
对照独立调度的完整记录、消息支持和独立根。一般区域 history/续接继续回归。
三步 AdamW、保留图切点、含 pending checkpoint 与换 owner 恢复均核对全部相关参数/槽。
standalone C++ 执行 literal quad、发送和平移不变的单原子 mean 导数；这不是 standalone 完整训练矩阵。

首个 Python 区域 VJP 原型 **46/2**：caps 与普通分配行共同求导导致 score 根的 None 变连接零。
按实际求解分支隔离后局部检查通过；没有按梯度数值屏蔽。
开发 tree 5d0e007b295767384c6c87e5795d7915194173da 为 **480/2**，C++ 构建和 standalone 通过；
新增 FP32 Python/native PDG 的质量根 input.1 相差 7.63e-6。关闭区域合批仍复现。
诊断定位为 Full 微小前向舍入在 Aggregate 商导数两个大项相减中放大。

修正保留原始 weighted mean 前向商，质量 VJP 用数学等价的 `dot(bar_u,z_i-u)/M`，
内容 VJP 为 `(m_i/M)*bar_u`；单一正质量原子的质量导数按恒等式返回连接零。
零总量的无 transform 根规则保持。独立单原子不变性、标量方向差分、既有均值 Jacobian
和 Aggregate 全部根/训练回归验证此表达。输入、dtype、前向数学函数和容差未改。
修正后的开发 tree 080ec009e52e469f08e9c7cecc1568c90668f333 为 **536/0**；
各失败的源码、原始日志和终态保留，不用新成功覆盖旧失败。
更早启动目录拒绝覆盖、缺少 grad-mode include、argsort 重载歧义均在 pytest 开始前失败，分别记录。

复现：检出固定实现提交，按 [development](../development.md) 构建，执行 JSON 中的测试列表，
使用 `--dtype both`、新的 basetemp 和当前 TIDE_BUILD_DIR；无需旧运行目录。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；单线程 ATen/BLAS、两个 build workers。
unit tide-core5-region-qualify-20261027 位于 background.slice，Nice=10，20 分钟上限。
原始输出 artifacts/jobs/core5-region-qualify，build manifest 已独立保存。
本报告不认证后续来源参数绑定、计算共享、混合/常驻后端、CUDA/NPU 真机或性能。
