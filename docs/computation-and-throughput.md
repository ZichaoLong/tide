# FP32 优先：规模、活跃量、吞吐与计算流程

本页回答实际训练的规模选择，以及为什么近参数、近 block 的 Tide 与 Transformer
仍有明显性能差异。训练优先 FP32，BF16 保留为次选；约定归属
[project scope](project-scope.md)，可运行设置见[训练建议](execution-recommendations.md)。

本轮新增的三种接收、FP32 百步训练及 A–D 对照在
[Reception 训练表](evidence/reception-training-20261118.md)；9B/17B 常驻首输出与缓存增量在
[连续推理表](evidence/reception-inference-20261118.md)。SSD 扫描公式、真实设备任务和
0.5B 双边成本归因见[性能报告](evidence/reception-performance-20261118.md)。
下文较早的全拓扑结果保留原身份，供拓扑与活跃量的横向参考。

这里整理各自冻结的已有证据。计算流程对应当前生产 tree `539c5eb1`，数学固定
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。下面每组数据保留原测量提交，
不能把较早实现的全拓扑矩阵称为当前 HEAD 重测，更不能把不同 B/L 排成同条件名次。

## 先区分四种“激活量”

| 量 | 本页含义 | 不能替代的量 |
|---|---|---|
| 总参数 | 去重后的唯一 Parameter 标量数，M=百万、B=十亿 | 每输入计算量 |
| 实际活跃参数 | 按实际作用的 owner/group/source 切片去重；Settle 每样本输入位置、Timed 每样本逻辑 tick，再取均值 | 非零梯度数、读取字节、FLOPs |
| Full 比例 | `full_actions / candidate_slots`；分母包括空接收槽 | 全部运算中被跳过的比例 |
| 激活/执行内存 | 中间张量、反向保存、私有状态与临时空间；表中 allocated 是更广的 allocator 峰值 | 纯激活字节、整卡实际使用量或空闲量 |

选中 Full 也不等于只执行这些行：矩形 query pack 的 padding、入边 padding 都可能
增加物理运算。共享权重按唯一参数只数一次，却仍需作用于多个不同输入。
完整口径由[比较契约](benchmark-contract.md)定义。

吞吐统一用**实际输入位置数除以完整训练秒数**；不把节点、padding 或逻辑 tick
当额外输入。完整训练含反向与 AdamW；Timed 含必要 flush。以下主体都是合成主干
训练，不包含下游任务头、数据流水线或任务质量评估。

## FP32 同尺寸拓扑与类型

单 NPU、B8/L128、两次预热后五次完整 AdamW 采样，报告中位数对应吞吐。
参数持续更新，每步私有状态清零；活跃量/Full 是初始 train 审计，不是五步轨迹均值。
源码 `778c65fb2961624bb416343eb3ad0f6f031deff7`；这是较早的统一矩阵。

| 配置 | 总参数 M | 实际活跃 M | Full/槽位 | 完整训练输入位置/s |
|---|---:|---:|---:|---:|
| FC-K | 9.334 | 4.170 | 29.645% | 8,259.56 |
| FC-S | 2.198 | 2.197 | 99.905% | 4,128.25 |
| HB Lattice-K | 5.759 | 5.592 | 90.779% | 3,041.88 |
| HB Exchange-X | 7.647 | 7.073 | 72.603% | 1,482.60 |
| HB Multiscale-X Settle | 15.694 | 14.450 | 70.645% | 767.09 |
| HB Multiscale-X Timed | 15.694 | 10.882 / tick | 52.227% | 225.53 |
| FC-K 149M | 149.207 | 66.470 | 29.550% | 5,265.62 |

Timed 累计触及为 **44.209M/实际输入位置**，可大于模型总参数，因为多个 tick 的
工作重复累计。若只以正接收候选为分母，它的 Full 比例为 71.452%，不能与上表
52.227% 混用。X 在当前绑定中是 D 节点用 K、H 节点用 S。

FC-S 参数较少，却在该点慢于 FC-K：两个配置的 Full 比例、记忆计算和消息活动都不同，
不能仅凭参数数目预测吞吐，也不能从此表单独归因 SSD 的耗时占比。
各模式、A–D、原样本、峰值和设备身份见[代表矩阵](evidence/roadmap-representative-20261118.md)。

## 扩大窗口与二十步训练参照

先看较早 FP32 长度搜索。同一组固定 B，改变 L；每点独立进程、一预热三采样，
每步私有状态清零。源码 `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`。

| 配置 | 总参数 M | B/L | 输入位置/s | allocated 峰值 GiB |
|---|---:|---|---:|---:|
| FC-K | 9.334 | 8/128 | 7,803.21 | 1.129 |
| FC-K | 9.334 | 8/256 | 12,214.85 | 2.106 |
| FC-K | 9.334 | 8/512 | 16,352.99 | 4.067 |
| FC-K | 9.334 | 8/1024 | 18,400.58 | 7.976 |
| HB Settle-X | 15.694 | 8/128 | 772.00 | 7.799 |
| HB Settle-X | 15.694 | 8/256 | 1,064.73 | 15.322 |
| HB Settle-X | 15.694 | 8/512 | 1,257.82 | 30.515 |
| HB-X 1B | 976.769 | 1/32 | 21.11 | 19.461 |
| HB-X 1B | 976.769 | 1/64 | 40.88 | 21.338 |
| HB-X 1B | 976.769 | 1/128 | 72.59 | 24.501 |

扩大 L 增加输入流/反向图范围，历史可见窗口不变。这些有限搜索先耗尽预设候选，
没有测出 OOM 或全局吞吐极限。HB 1B 的 L128 初始活跃量为 277.432M/输入位置，
四次更新后为 254.190M。[原始搜索与四类对照](evidence/capacity-search-20261118.md)
保留每点活动变化、波动和完整命令。

上一组二十步训练组合使用 `region_batch_size=4`、`checkpoint_reception=true`、
`ssd_summary_backend=prefix`，整区域重算关闭。下表是真实保留历史的 20 次更新，
吞吐按更新 3–20 的总输入位置/总 `complete_seconds`；峰值取全部 20 步。

| 配置 | 精度 | 总参数 M | B/L | 稳态输入位置/s | allocated/reserved GiB |
|---|---|---:|---|---:|---:|
| HB Settle-X | FP32 | 15.694 | 24/256 | 2,074.50 | 38.285 / 42.998 |
| HB Timed-X | FP32 | 15.694 | 12/128 | 317.40 | 41.653 / 46.387 |
| HB-X 0.5B | FP32 | 494.111 | 1/128 | 96.08 | 15.372 / 16.506 |
| HB-X 0.5B，次选参考 | BF16 | 494.111 | 1/128 | 90.61 | 13.180 / 14.641 |

Settle/Timed 来源 `22527cf472fc4d6ea9ca7ce22528b7e4ca9cc96a`；0.5B 来源
`ae247dee82c60b63495e9abea7f703abd84ac3c7`。这 20 步没有逐步有效触及均值审计，
所以不把另一零态负载的活跃量填进该表。三个 FP32 点在各自保存点的配对恢复更新
均在原阈值内；独立进程也实际完成 20→21 更新，后者没有原 owner 对照，不是额外
轨迹等价通过。[连续记录](evidence/training-continuous-20261118.md)保留原数值限制。

这组较早的 FC 大 batch 连续训练点为 **BF16** B128/L256、38,248.5 输入位置/s，
保留为次选参考。本轮已另完成 FP32 B128/L256 的100更新：30,333.2输入位置/s，
allocated/reserved 为37.36/39.58GiB，配对恢复更新阈内；见[新训练证据](evidence/reception-training-20261118.md)。
两条记录的源码、精度及更新步数不同，不把其比值称为纯精度切换收益。

本机每卡 64 GiB；48 GiB 是实验的 sampled allocated 预算，不含 reserved 和 Torch 外
占用，也非硬显存隔离。较新准入快照的设备外占用约 2.8–3.3 GiB，不是运行峰值。
加入任务头、数据缓存后的余量仍应从实际设备观察，不能简单用 64−allocated 推算。

## 0.5B、1B 与多卡容量

以下四点是同 B1/L8 的较早 FP32 测量，一预热三采样、零私有状态、完整 AdamW。
源码 `e2a4d39104a006feec4525e534eadae4a94cde0d`。全部为 independent 参数。
活跃量与 Full 比例取初始 train 审计，不是采样期间的均值。

| 配置 | 总参数 M | 活跃 M | 节点/层数、d | Full/槽位 | 输入位置/s | allocated GiB |
|---|---:|---:|---|---:|---:|---:|
| FC-K 0.5B | 499.342 | 169.814 | 192/12、384 | 16.862% | 22.447 | 9.305 |
| FC-K 1B | 998.536 | 336.900 | 384/24、384 | 16.536% | 9.215 | 18.616 |
| HB-X 0.5B | 494.111 | 148.267 | 1712/9、128 | 19.721% | 2.930 | 9.208 |
| HB-X 1B | 976.769 | 279.603 | 848/5、256 | 18.499% | 5.706 | 18.200 |

这张短窗口表主要暴露参数持有、调度与小矩阵成本，不代表充分扩大 B/L 后的训练速度。
0.5B HB 比 1B HB 窄且节点更多，不能从两行排序推断“模型越大越快”。0.5B 的上一组
B1/L128 连续训练为上表96.08，本轮B4/L128百步为267.40；B/L、状态和实现都变了，
不能把比值全归给某个优化。
详情及 A–D 见[规模记录](evidence/roadmap-scale-small-20261118.md)。

9B/17B 也有实际 FP32 更新，但目前只是两步容量记录：HB Multiscale-X、d768、
Settle、independent、B1/L8、host-stage 传输、区域 serial、AdamW no-foreach，无预热。

| 实际参数 | NPU 数 | 两次完整更新 s | 16 输入/两步总秒数 | 最高单卡 allocated/reserved GiB |
|---|---:|---|---:|---:|
| 8.778B | 4 | 3.859561 / 2.058335 | 2.704/s | 38.489 / 42.480 |
| 16.707B | 6 | 5.761769 / 2.508131 | 1.935/s | 48.594 / 54.379 |

分别来源 `36c42a7a3ef765fe6257c42c7c1404d28857b77c`、
`8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee`。状态 0→8→16，optimizer step 1→2，
完整边界包含跨卡传输、detach、同步和阶段记录 IO。这两条训练记录没有测有效触及
参数，也没有该大模型的 checkpoint/恢复资格。不能拿不同 seed 的单卡/CPU 推理
活跃量填入，更不能称为稳态吞吐或区域并发加速。
[区域容量证据](evidence/region-capacity-20261118.md)保留单卡峰值、放置和全部限制。

## Full 稀疏比例为什么不直接变成加速

人工支持压力用例：FC-K n32×4、d128，总参数 36.892M，FP32、B4/L32，
源码 `e2a4d391…`。评分初始化被人工控制，强度预算仍合法；这些不是训练获得的稀疏性。
活跃量、Full 比例和 packing 比取初始 train 审计。

| 用例 | Full/槽位 | 活跃 M | pack 执行/选中 | 输入位置/s | 三样本 range/median |
|---|---:|---:|---:|---:|---:|
| natural | 9.552% | 10.336 | 5.449 | 569.22 | 14.68% |
| 每区 1/32 | 3.125% | 8.449 | 1.000 | 916.31 | 100.20% |
| 每区 3/32 | 9.375% | 10.284 | 1.000 | 1,271.94 | 4.61% |
| 全 Full | 100% | 36.892 | 1.000 | 1,338.62 | 1.41% |

Full 仅 3.125% 时，仍触及约 **22.9% 总参数**。每行都有 16,384 个候选准备槽和
1,048,576 个正区域注意力对。Full 之前的工作还在，natural 的不均匀选择还增加 padding。
1/32 那行原始秒数为 0.139690、0.257237、0.117262，明显不稳定。不同人工初始化还
改变分数、分配与后续轨迹，因此不能据此断言全 Full 的数学总是更省，或稀疏必然慢。
[完整压力证据](evidence/roadmap-supplementary-20261118.md)保留原样本。

## 实际计算流程：先准备，再分配，最后 Full

当前完整模型是 PyTorch；LibTorch 是另外保留的区域评分切片。B、时间和节点已张量化，
不是在 Python 中逐节点/逐 token 调一次小模型。主循环仍按依赖层、相容区域批和
固定长度块组织。[model.py](../src/tide/model.py) 的核心顺序可写为：

```python
T = L if settle else L * period
for layer in topological_layers:
    for regions in compatible_region_batches(layer):
        msg, mass, edge_class = gather_predecessors()  # [N, B, T, Emax, d]
        received, u = reception(msg, mass, edge_class)
        write_mass = minimum(1, received)             # FC 可达状态下恒为 1

        desc, context, next_memory = memory.prepare(
            u, write_mass, original_int64_times, old_memory)
        # 正接收的记忆更新不以稍后是否选中 Full 为条件

        if fixed_through_hub_region:
            control = minimum(1, 2 * received)
        else:
            score = describe_and_score_within_each_region(u, received, desc)
            control = allocation(score, received)    # 独立的区域约束

        full = memory.finish(context, control)       # 打包 control > 0
        publish(content=full, strength=control)      # 保留独立区域/节点私有状态
```

Timed 输入每 P 个逻辑 tick 注入一次，所有边传播延迟一 tick，默认 P=4；当前 evaluator
仍整段处理长度 `L*P` 的张量。Settle 同一输入位置内按层完成传播，时间轴长度为 L。
compact-9 Timed 的有限窗口尾部要 flush `ceil((9−1)/4)=2` 个输入周期，即 **8 tick**。
时间轴、准备工作、历史容量和尾部都不同，不能只把它理解为 Settle 换一种调度顺序。

### 1. 逐边接收

设节点 v 收到内容 y_e、非负强度 a_e、边类别 k_e。默认 `source_residual` 为：

\[
N(x)=\frac{x}{\sqrt{\operatorname{mean}(x^2)+10^{-5}}},\qquad
g_{v,e}=y_e+B_v\operatorname{SiLU}(A_vN(y_e)+e_{v,k_e}),
\]
\[
r_v=\sum_e a_e,\qquad
u_v=\begin{cases}\displaystyle\frac{\sum_e a_e g_{v,e}}{r_v},&r_v>0,\\0,&r_v=0.\end{cases}
\]

这里的分母是 r，不是 `max(1,r)`。默认接收隐层为 d/8，每条执行边的两次基础矩阵
乘约 d²/4 个 multiply-add，尚不计 adapter、norm、加权求和。非线性在聚合之前，
不能直接替换成“先平均消息，再做一次 MLP”。三种 `incoming_*` 候选另有入边间加权
self-attention；默认 `source_residual` 没有这项入度平方计算。

实际张量按每区最大入度 Emax 补齐；零质量遮罩保护数学含义，并不保证所有接收投影
都跳过这些零边。见 [reception.py](../src/tide/reception.py)。

三种 incoming 绑定在上述逐边 MLP 之前，对同一接收节点、同一样本、同一接收逻辑时刻θ的
入边做 self-attention。令 \(x_e=N(y_e)+e^{SA}_{v,k_e}\)，对每个宽32的头：

\[
q_e=Q_vx_e,\quad \kappa_e=K_vx_e,\quad \upsilon_e=V_vx_e,\qquad
h_e=\frac{\sum_f a_f\exp(q_e^\top \kappa_f/\sqrt{32})\upsilon_f}
          {\sum_f a_f\exp(q_e^\top \kappa_f/\sqrt{32})},
\]
\[
\widetilde y_e=y_e+O_v\operatorname{ConcatHeads}(h_e).
\]

随后以 \(\widetilde y_e\) 代替第一式的 \(y_e\)，再做逐边残差 MLP 和按 \(a_e\)
的最终加权聚合。空分母使用声明的零约定；light64/light128 的 Attention 总宽为64/128，
逐边 MLP 仍为d/8，wide的两者宽度均为d。接收 Attention 轴为入边，区域 Attention
轴为同区节点，KV Attention 轴为历史原时刻，这三种形状不能混成一个站点。

```python
# 保留接收节点 × sample × time 的独立批量轴，Attention只混入边
x = norm(messages) + sa_source_embedding
q, k, v = Q_v(x), K_v(x), V_v(x)
mixed = weighted_self_attention(q, k, v, prior=edge_strength)
y = messages + O_v(concat_heads(mixed))
messages = y + B_v(silu(A_v(norm(y)) + mlp_source_embedding))
received, content = weighted_edge_mean(messages, edge_strength)
```

Q/K/V/O/A/B是各用途独立的有效节点矩阵；共享时由共享基座及各自私有适配器组成。
SA与MLP来源嵌入也是两套，均按声明的参数归属持有。

融合QKV减少投影调用，但当前每次还会拼接权重；结构mask仅排除不存在的边，真实
零质量边的导数域保留。二者都是同一接收数学内的执行选项，不能自动改选Agg。
[接收执行说明](reception-execution.md)给出接口和零质量边界，
[实测](evidence/reception-performance-20261118.md)保留融合变慢或峰值增大的形状。

### 2. K 记忆：未选中 Full 也要写 KV 和算描述

准备阶段对候选计算 `KV(N(u))` 和 `p=W_pN(u)`（p 宽 d/4），使用原始 int64 时刻
合并旧 KV、判定可见窗口、计算长短摘要并压紧缓存。两类摘要概括为：

\[
\omega^{long}_{tj}=w_j\mathbf1[0\le t-t_j<W],\quad
\omega^{short}_{tj}=\omega^{long}_{tj}e^{-\gamma(t-t_j)},\qquad
m^{a}_t=\frac{\sum_j\omega^a_{tj}p_j}{\max(1,\sum_j\omega^a_{tj})}.
\]

该摘要分母与接收分母不同。Full 选中以后才生成 Q、施加 RoPE、读取 KV：

\[
\alpha_{tj}=\frac{w_j\mathbf1_{visible}\exp(q_t^\top k_j/\sqrt{32})}
{\sum_s w_s\mathbf1_{visible}\exp(q_t^\top k_s/\sqrt{32})},\qquad
z_t=\min(1,\sum_{visible}w_j)\sum_j\alpha_{tj}v_j,
\]

随后输出投影、残差与 Full FFN；空可见质量按声明的零行处理。默认逻辑窗口 W=128P，
Settle 的写入相隔 P tick，所以缓存 128 槽；Timed 可以每 tick 写入，默认缓存 512 槽。
可见性始终按原始时间判断，不按压紧后的查询序号重算。

已有 query chunk、原时间窗口 tile、NPU 多头 Q 融合和线性缓存压紧；不能把逐头 Q
调用或 NPU 时间排序当成当前实现必有的瓶颈。见 [kv.py](../src/tide/kv.py)。

### 3. S 记忆：矩阵递推/读出在选择之前

输入投影后对内容和 B/C 做四 tap 原逻辑时间卷积；当前项未乘质量，历史缓存项为
质量加权投影，缺失原时刻补零。卷积后经过 SiLU，所得量记为 ν、B、C。
对每个宽 32、状态宽 32 的头 h：

\[
\delta_{h,t}=w_t\operatorname{softplus}(\widehat\Delta_{h,t}+b_h),\quad
a_{h,t}=\exp(-e^{A_h^{log}}\delta_{h,t}),
\]
\[
H_{h,t}=a_{h,t}H_{h,t-1}+(\delta_{h,t}\nu_{h,t})B_t^\top,\qquad
R_{h,t}=H_{h,t}C_t.
\]

w=0 时矩阵不写入也不衰减。描述网络需要 R，因此不能等 Full 选完后才做矩阵扫描。
选后才完成 gate/skip、输出投影和 FFN：

\[
g=[R_h+D_h\nu_h]_h\odot\operatorname{SiLU}(z^o),\qquad
x=u+O(\rho\odot N(g)),\qquad Full=FFN(x).
\]

实现没有保存每时刻完整 H，而是长度至多 32 的块扫描。省略头和 batch 轴，令
V 的第 j 行为 \(\delta_j\nu_j\)，块内的结果是：

\[
P_{ij}=\mathbf1_{j\le i}\prod_{k=j+1}^{i}a_k,\qquad
Y=(CB^\top\odot P)V+\operatorname{diag}(\prod_{k\le i}a_k)CH_{in}^\top.
\]

块间只传递边界矩阵状态，P 用直接乘积构造，保持零衰减合法。新的 `prefix` 优化
只作用于短/长**向量摘要**，利用仿射合成：

\[
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\ b_2+a_2b_1).
\]

它对长度 T 的时间轴用 ⌈log₂T⌉ 个批量合成阶段，避免摘要也生成块内成对传播矩阵；
Timed 的 T=LP，没有替换完整矩阵 SSD。

本轮另加入 `ssd_scan_backend=incremental/doubling`：L=1直接执行上面的矩阵递推，
省去块内成对传播；doubling 对长块的乘积P使用有界乘法树，替换原block里的cumprod。
不使用逆衰减或log差，保留零衰减导数及块边界矩阵状态；其反向保存可能更多。
NPU L256 基元前向约32.32→5.03ms，完整小模型FP32训练吞吐约+19.25%，两种
计时边界分别保留，不能把局部倍数套给全模型。公式、全部根VJP与设备任务见
[SSD证据](evidence/reception-performance-20261118.md)。
见 [ssd.py](../src/tide/ssd.py)、[ssd_memory.py](../src/tide/ssd_memory.py)。

### 4. 区域描述、Attention 与强度分配

描述把 `[N(u), log(1+r), m_s, m_l, log(1+W_s), log(1+W_l), p*m_s, p*m_l]`
共 2d+3 维编码到 d/2。K 的 m_s/m_l 是前面的短/长摘要；S 的
\(m_s=Z_s/\max(1,W_s)\)、\(m_l=W_{read}R\)，后者是矩阵读出投影，
不是 KV 式长摘要；描述仍保留短/长权重 W_s/W_l。
区域 Attention 把**节点作为序列、B/时间作为 batch**，
prior 为 w；后接区域 MLP，得到 s_v。这些工作都先于 Full 选择。
当前实现已将末端只供标量消费的 `aᵀW₂h` 收缩为 `(W₂ᵀa)ᵀh`，避免生成无用向量。

以当前常用 quad 为例：

\[
w_v=\min(1,r_v),\quad\mu_v=\min(1,2r_v),\quad
u_v^{cap}=\mu_v,\quad\ell_v=0\ \text{或有界 H 的 }\mu_v/4,
\]
\[
U=\min(k,\sum_vu_v^{cap}),\qquad
L=\max\left(\sum_v\ell_v,\ \min(\min(1,\sum_vw_v),U)\right),
\]
\[
\max_c\sum_v(s_vc_v-\tfrac12c_v^2),\quad
\ell_v\le c_v\le u_v^{cap},\quad L\le\sum_vc_v\le U.
\]

其解为 \(c_v=\operatorname{clip}(s_v-\lambda,\ell_v,u_v^{cap})\)。若 λ=0 的总量
已经合法，就不收紧到边界；否则解对应总量。quad 当前用断点排序和有限重构，特殊
数值情形回退二分。Shannon/pool 有各自契约，不应套用该式。

**k 不是 top-k 节点数量。** 四个节点评分均为 1、上界均为 1、k=1 时，可得
`c=[0.25,0.25,0.25,0.25]`，四个节点全部 Full。很小的正 c 仍要求完整局部算子，
不能按 c 的数值把 FFN 算成四分之一。见 [region.py](../src/tide/region.py)、
[allocation.py](../src/tide/allocation.py)；固定数学见
[区域分配](../math/local-semantics/weighted-message/region-allocation-learning-note.md)。

### 5. Full packing、共享参数与合批

一个 pack 中设 `q[n,b]` 是选中查询数，`N_active` 是至少有一个查询的节点数：

\[
Q_{max}=\max_{n,b}q[n,b],\quad
\text{selected}=\sum_{n,b}q[n,b],\quad
\text{executed}=N_{active}\,B\,Q_{max}.
\]

有效 mask 恢复正确输出，矩形计算仍可能经过 padding。K 按原时间 tile 分别打包，
总 executed 要跨 pack 累加。Full 的 gated FFN 是两支 d→4d、一支 4d→d，
仅基础矩阵就约 **12d² multiply-add/执行行**，不含适配器和反向。

共享模式的矩阵是 \(W_v=W^0_{g(v)}+L_vR_v\)。当前计算为
\(W^0_{g(v)}x_v+L_v(R_vx_v)\)。当本次 owners 的静态可能集合全部属于同一共享组时，
基础矩阵合并节点/B/时间做一次大 mm；混合组调用走 bmm。
私有适配器保持两次乘法，不物化各节点稠密矩阵。共享减少参数存储，输入不同意味着
矩阵作用次数没有消失。见 [parameters.py](../src/tide/parameters.py)、
[layout.py](../src/tide/layout.py)。

相容区域合批仍保留独立区域轴；每区自己的 Attention、分配约束和私有状态不变，
不是把 r 个区域混成 r×n 节点的单一区域。合批减少调用，但可能扩大 Qmax 或别的
补齐尺寸，故“调用变少”和“MAC 变多”可以同时发生。

## 与 Transformer 的差距：计算和测量分别解释

常规 Transformer block 的主要流程是：

```python
q, k, v = fused_qkv(learned_rmsnorm(x))
q, k = rope(q), rope(k)
x = x + output(native_or_flash_window_attention(q, k, v))
gate, up = fused_gate_up(learned_rmsnorm(x))
x = x + down(silu(gate) * up)
```

本仓库基线已经使用完整 B/L/head 张量计算、融合 QKV、融合 gate/up、缓存 RoPE/窗口
mask 和 native unit-prior Attention。Tide 还需逐边非线性、选择前记忆/摘要、区域评分、
约束分配、动态 gather/pack/scatter；这些既有数学额外工作，也有实现可优化成本。
训练反向会经过选择前的这些路径，不只是选中 Full 的 FFN。

A/B 分别为近节点 block 数×近总/实际活跃参数，C/D 为近层数深度×近总/实际活跃参数。
合法合批是 Tide 的结构优势，不应为了比较禁用；四类匹配都没有保证逐边工作、矩阵形状、
算术量、历史访问或关键路径相同。Transformer 也不是已经证明的全局最优。

上一组 FP32 零态完整训练对照（源码 `2c011b8a…`，两预热五采样、Transformer chunk256）：

| 配置 | B/L | Tide 秒 | A 秒 | B 秒 | C 秒 | D 秒 |
|---|---|---:|---:|---:|---:|---:|
| Settle-X | 24/256 | 2.857489 | 0.598656 | 0.591192 | 0.081500 | 0.055614 |
| Timed-X | 12/128 | 4.238573 | 1.214859 | 1.210969 | 0.058430 | 0.058318 |

Settle 对 A/B 耗时约 4.77/4.83 倍，Timed 约 3.49/3.50 倍。另一方面，较早 0.5B/1B
短窗口矩阵中 Tide 确实快于 A/B；HB 1B FP32 B1/L128 的 Tide 用时分别是 A/B 的
**0.639/0.646 倍**。
不同规模/形态并非同一个性能结论；C/D 少而宽的 block 仍明显更快。
全部构造器、匹配偏差和模式边界见[模式主表](evidence/training-modes-20261118.md)及
[长度搜索](evidence/capacity-search-20261118.md)。

### 已测成本证据

本轮已经补齐小型FP32与0.5B FP32的双方同步阶段诊断和真实Ascend设备trace。
0.5B B1/L128的Tide普通完整更新1.3437s；另一次侵入式前向0.7773s中，allocation
为0.2153s（该诊断前向27.69%）、gather约69.8ms、packing约64.5ms、SSD矩阵约36.6ms。
其K查询执行行约为选中行5.25倍。Transformer A有1712串行层，trace硬件首末跨度
14.96s而kernel区间并集1.78s，支持存在显著调度间隙；这些不是正常时间的可加占比。
[本轮性能归因](evidence/reception-performance-20261118.md)保存具体形状、原样本、
双方operator、source和设备身份；不同规模的A/B与Tide胜负确实不同。

下表是较早的 **Settle-X BF16 B8/L256** 机制见证，188节点、9层、15.694M
参数；它解释计算机制，不能充当 FP32 成本百分比。来源分别为 `d5c7f30b`、`ae247dee`，
Transformer 候选为 `007a3828`。正常时间与侵入式诊断严格分开。

| Tide 候选 | 正常训练秒 | allocated GiB | dispatch 调用 | NodeLinear 前向 multiply-add G |
|---|---:|---:|---:|---:|
| 单区域调用 | 1.9715 | 13.521 | 94,554 | 122.158 |
| 仅相容区域 batch4 | 1.5330 | 13.652 | 60,393 | 123.383 |
| batch4 + prefix + Reception 重算 | 1.4707 | 10.213 | 59,522 | 123.464 |

不同候选预热后活动轨迹已有差别，上表是该尺寸下的有限观察，不是严格同轨迹纯加速。
消息 slot 为 2,850,816，即每实际输入位置 **1,392 条执行边槽**。
serial 有 270,583 次 Full，约占 candidate slots 的 70.28%；K 的 147,703 selected
扩成 200,704 executed。batch4 的 K executed 增为 205,568，但调用减少，正常时间下降。

同负载 Transformer A/C 的前向 Linear 约 32G multiply-add；Tide 仅 NodeLinear 就
约 123G，约为其 3.8 倍，且还没计 Attention、普通 regional Linear、SSD、归一化或反向。
这不是完整 FLOPs 统计，也不是“理论必慢 3.8 倍”，却足以否定“近参数就应近计算量”。

另外一次同步侵入式观察中，该较早组合的矩阵 SSD 为 0.25349s、向量摘要 prefix 为
0.01018s、allocation 为 0.09314s、messages.gather 为 0.06982s。
反向 0.64914s 内含 Reception 重算 0.13345s，不能再把重算加一次。
实际设备 trace 的 128 次 Cumprod 累计 233.860ms，任务类型是 **KERNEL_AICPU**，
属于设备任务而非主机 CPU fallback；8 次 ScatterElements 累计 63.874ms。

这些不同诊断的时间有同步、嵌套或重叠，不能相加后除以正常训练 1.4707s，声称各项
占比。当前没有跨所有 FP32 拓扑建立严格可加的关键路径百分比。
[成本原证据](evidence/training-costs-20261118.md)给出观察边界和原始 Ascend trace。

### 分块与重算已经用了什么，还能改什么

当前 K 有窗口/query tile 和 native/fusion/tiled Attention；SSD 有 32 长度块与边界
矩阵 carry，向量摘要另有 prefix，矩阵传播可选doubling、单位置用直接递推；
训练还有 Reception 局部重算及可选整区域重算。
这些避免某些大中间张量，却不能像 Attention 那样自动消除逐边接收或区域求解。
Flash 类优化的前提是相应数学、prior/零质量导数、mask 和后端支持，而非任何矩阵都
可以用一个 Flash Attention 调用替代。

重算通常增加反向工作；同 B/L 下若完整时间增加 74–81%，吞吐会变为原来的约
55–57%（下降约 43–45%），不是吞吐也下降 74–81%。已有那组整区域重算结果还需按
原候选区分是否同时改了局部 Attention，不能将全部差值归给重算。局部 Reception
重算则保留了成本更高的其他前向路径，在上述组合中观察到明显内存收益。

更低峰值可能允许更大 B/L，最终提高单位时间训练输入；因此选型看同资源预算的完整
训练吞吐与实际余量，而非只看一次 forward 或同尺寸峰值。已有候选都保留。
Transformer 自身 query chunk128→256 也使 A/B 在成本实验中减少约 19–21% 时间，
而提前 contiguous 没有一致收益，说明双方批量粒度都需要实测。

矩阵扫描的doubling优化与单位置直接递推已在本轮实施并测量；可选static_view也已减少适用静态
owner的参数物化，9B持续decode观察到约9.5%吞吐收益而峰allocated基本不变。
剩余明确方向是分配的小算子/同步、按有效查询行分桶以减少padding，以及索引梯度归并。
分桶会增加launch，专用分配反向要保留解面和零质量导数，不能仅由trace推算净收益。
这些方向仍需独立公式、全部梯度及续接检查，不是已证明没有优化空间。具体机会归属
[ROADMAP](ROADMAP.md)；各候选继续保留，未做未经测量的全局默认替换。

## 复核入口与限制

- 参数/模式/采样命令、源码与文件 SHA：以上各份 `docs/evidence/*.md` 的同名 `.json.gz` 审阅证据。
- 生产实现定位：[architecture](architecture.md)、[semantic map](semantic-map.md)。
- 实际起步/覆盖/恢复：[execution recommendations](execution-recommendations.md)。
- 数值通过与保留超差：[precision](precision.md)、[training paths](evidence/training-paths-20261118.md)。

所有列出的 NPU 数据使用各自记录的协作队列独占准入，但主机和队列外任务不受本项目
控制；跨卡/跨时间观察保留原身份，不是全机独占或同时配对。CPU FP64 是诊断工具；
CUDA、全模型 LibTorch、peer DMA/HCCL 和 compiled/resident 训练没有因此获得资格。
本页只增加解释和可发现性，没有修改数值阈值、生产实现或原始测量记录。
