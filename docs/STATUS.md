# 当前交接：训练默认、路径核对与性能归因进行中

用户已确认执行 [ROADMAP](ROADMAP.md) 的新六项计划。起点为
`3a49a789a252ffc6eac215b612e9407fdb7428c9`，分支 `graph-execution-foundation`。
数学仍固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，本轮108文件核验通过。
持续 FC/HB 实例路线不变，通用基座未启动。计划中的能力不视为已验证。

## 当前工作与资源

- 主代理负责集成、预算搜索、生产优化、统一设备准入和证据；三个独立工作树分别实现
  路径诊断、双边性能归因、公开训练入口。代理不独立提交 NPU 任务。
- 起步无遗留 Tide 运行服务；已核对账户83.4 GiB使用/300 GiB软配额/700 GiB硬配额，
  主机约1.8 TiB可用内存。NPU有其他项目任务，后续按协作队列逐作业准入。
- 本轮尚未取得新的实验结论。关键缺口是 batch-only 同预算搜索、不同实现首处分歧、
  双边成本说明、选定配置二十步连续训练及新进程恢复。默认设置待证据决定。
- 四份原用户未跟踪文档保留；原始hash在
  [前阶段集成记录](evidence/capacity-validation-20261118.json)。不改参考仓库、公共环境或下游。

## 可复用的当前实现与资格

完整模型是 PyTorch，C++17/LibTorch 为独立区域评分切片。已实现合法 FC/HB 主菜单、
可选 Hub/时钟、相容区域合批、区域重算、Attention 后端选择、完整区域归属及
serial/per_device 执行；区域放置目前要求 batch1。详见 [实例](instances.md)、
[配置](optional-bindings.md)、[放置](placement.md) 和 [native](native-regional.md)。

前阶段交付见 [集成验证](evidence/capacity-validation-20261118.md)：冻结 `8a22530d`
完整CPU544项通过，`5ae060db`定向17项通过。硬件实验各有实际冻结提交，不能自动
覆盖本轮修改。CPU/NPU有真机证据；CUDA、peer DMA/HCCL、编译训练仍无资格。

[容量搜索](evidence/capacity-search-20261118.md)全部到有限候选停止，未证明硬件极限。
[执行候选](evidence/capacity-execution-20261118.md)中 batch4 同尺寸用时为基线
0.695–0.792、峰值约1.009–1.038；重算增加用时但明显降低峰值。旧 combined 同时
改变local_math、合批、重算，不能当纯合批重算结果。32 GiB搜索缺少batch-only。

[0.5B/1B连续](evidence/capacity-continuous-20261118.md)已有三模式、三段及新进程恢复；
[区域多卡](evidence/region-capacity-20261118.md)已有小型资格、六更新观察及9B/17B两步
容量事实，不是大模型长序列长期训练资格。并发后续活动不同，速度比不是严格同轨迹加速。

[舍入](evidence/capacity-rounding-20261118.md)修复了BF16偏置提升错误，并保留全部有限
支持/时间/梯度/更新超差。[隔离安装](evidence/capacity-installed-20261118.md)保存数据
和下一前向精确，但placed BF16恢复梯度/更新有超差；同owner并发反向可变。已有小见证
不能解释所有轨迹。新轮将有界核对，原阈值与失败不改写。

完整结果从 [证据索引](evidence/README.md) 查阅；日常复用
[性能经验](performance-lessons.md)，不需要重跑旧矩阵或查看退役历史。
