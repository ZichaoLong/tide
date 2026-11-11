# v8 直接实现的范围与数学对应

源码位于`tide_lab/direct_semantic*.py`，独立依据固定教材编写；没有复制基座执行核心。
`scripts/resolve_direct_v8.py`只调用已固定安装的公开规划接口导出图、局部配置与来源类别。
执行源码不import基座。它是本任务固定FC/HB的整段训练函数，尚不具备公共continuation/完整返回协议。

| 组件 | 实现及独立检查 |
| --- | --- |
| 聚合/共享 | 接收权重先归一化后混合内容及source变换；region/node/source/adapted按独立参数或区域基座+低秩绑定。逐来源标量和及共享权重VJP核对 |
| KV | 区域共享KV基座+节点低秩适配；逻辑时间窗口、sum/qry、短/长量、RoPE、距离bias、带权多头Full；独立小型显式历史求和检查 |
| affine/delta | 两个时间尺度的有效矩阵；保留、写入、当前query分别驱动；展开历史的结果和VJP与独立逐步矩阵递推比较 |
| Gated Delta | 固定逻辑时间卷积、q/k L2、标量保留、delta误差写入、逐头RMS后输出门；使用同一经过验证的delta创新量展开 |
| SSD | 分组B/C、标量保留、w·softplus写入、直通D、先输出门后整体RMS；使用历史展开 |
| Describe | cat/match、ind/att、content/content_stats、cat/grouped证据、三种Read，按教材独立参数；Read不会改变Full所读的新状态 |
| 发送 | quad/bern/Shannon/ent1.5/ent2、三类约束、硬排序、三种Attention和proportional；约束函数使用最优解隐式VJP，不对求根迭代链自动求导 |

每条边必须满足delay=目标region rank−来源region rank，才能在同token位置按rank顺序联合执行。
卷积间隔仍为原始逻辑时间；缺事件占零槽，不能把最近有效接收挤到相邻token。
私有有效状态每个独立文档块重新开始；不提供跨chunk原始状态恢复。应用checkpoint只在完整optimizer边界保存。

delta创新量使用严格下三角矩阵T：`(I+T)e=beta*v`，
以有限幂倍增计算其精确实数多项式逆；不使用CPU回退的三角求解器，不除以可能极小的累计gate。
它是独立直接实现的数值路径。小型FP64前向/VJP检查支持该公式，不能替代所有规模与精度的等价资格。
正式训练前还要每项实际形状CPU/NPU、因果、共享optimizer及新进程恢复检查。

全部候选更新记忆，包括未发送节点。Full物理上对整个固定图计算，然后用`c>0`遮罩，
零支持的内容及该边界反向均为零。`work/full`表示逻辑选择数，`work/physical_full_rows`表示实际Full行数；
后者包括无候选行。这种策略便于筛选数学候选，不支持真实稀疏计算或其性能结论。
本任务只有单位外部输入及正发送原子，候选存在性可由接收强度判断；
更一般“存在零强度原子”和“没有原子”的公共接口尚未实现，不能把该限定检查推广到任意输入纤维。

正指数按当前dtype的normal-tiny下限求值，零导数用于下限以下；掩码对应的数学零保留。
这是显式有限精度探索策略，**不是严格实数等价**，不能把下溢零计作学习稀疏。
观测`numerical/positive_floor_events`计被下限保护的已求值指数/分配坐标（时间核可被批样本共享，
不是独立样本或激活计数）。保留原1e-6/1e-5严格比较及未知原因；可行性残差>2e-5或非有限停止对应路径。
排序同分用固定节点序；截断取饱和侧导数，单点全上限可行集沿mu，等上下界只向同一个B输入反传一次。
Attention零基权重采用零边界VJP；这是声明的边界约定，不等同于所有单侧方向导数。

局部初始化使用`role-keyed-fan-in-v1`：seed+参数角色名+形状固定随机流，
同形公共角色不会因为另一模块增加参数而更换初始张量。embedding std0.02，普通矩阵fan-in，
KV低秩输出额外0.1，source适配左因子0.01；bias零、输出RMS/SSD skip为1。
不继承旧checkpoint，不覆盖原std0.15失败。不同族参数量和连接梯度元素分别记入检查表。

开发检查发现的排序API关键字错误及FP64时间核先算FP32的问题在启动正式局部任务前修复；
原失败输出保留于会话及开发记录。最初不完整规划表保留`.local/v8-resolve-incomplete-r1`，不作为可运行候选表。
