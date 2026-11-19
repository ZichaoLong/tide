# 记忆与参数绑定候选

本页定义固定快照之外的显式 FC/HB 实例扩展。数学固定
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，`math/` 原字节不变。
反向混合、窄 KV、分模块共享改变函数或参数化，不是旧数学的等价实现优化。
资格从 [STATUS](STATUS.md) 和固定源码证据读取；现有默认和有价值实现保留，默认入口主要服务训练。

## 五类与形状

| 候选 | 普通 D | Hub H | 新增变化 |
|---|---|---|---|
| X | KV | SSD | 参照 |
| S | SSD | SSD | 普通节点记忆函数 |
| X_reverse | SSD | KV | 两种角色的记忆函数 |
| X + kv_width=m | 窄 KV | SSD | Q/K/V 总内部宽共同变为 m |
| X + memory_sharing=region_shared | KV | SSD | 仅记忆区域基座＋节点私有适配器 |

没有 Hub 时 X 等于 K，X_reverse 等于 S。`kv_width` 只作用于 K 节点；全 SSD
或没有 K 节点的图拒绝这个未使用选项。m 为正的 32 倍数且不超过内容宽 d。
首轮 d=768 的有限 m 为 384/256/128；内容 d、摘要 s=d/4、描述和区域尺寸不变。
Q/K/V 头数都变为 m/32，头维保持 32；**不是保留 Q 头数的 GQA**，不与 GQA 合并计数。

`ffn_width=f` 默认 4d，允许正的 16 倍数，只改变 Full 的 gate/up/down，区域 FFN 不变。
固定图比较保持 f 不变；近总参数比较可以调整 f，但必须声明这个额外变化。
总参数相近不意味着记忆/FFN 配比相同。SSD 不受 kv_width 影响，保留 d/32 个 32×32 状态头。

## 窄 KV 局部公式

令 z=RMSNorm(u)，内容 u 为 `[N,B,L,d]`，原始逻辑时间 t 为 int64：

\[
[K,V]=W^{KV}_v z,\quad W^{KV}_v\in\mathbb R^{2m\times d},\qquad
Q_h=W^Q_{v,h}z,\quad W^Q_{v,h}\in\mathbb R^{32\times d},\quad h<m/32.
\]

摘要投影仍为 `s×d`，当前特征及短/长摘要的原公式不变。缓存 K/V 为 `[N,B,T,m]`；
查询/读取按 `[N,B,m/32,Q,32]` 计算。先在 int64 域求相对时间，再执行原 RoPE。
只有正接收写入；可见性是 `0 <= t_query-t_key < window`，不能用稀疏事件序号替代时间。

\[
r_h=\operatorname{Attention}(\operatorname{RoPE}(Q_h),\operatorname{RoPE}(K_h),V_h;w,\mathrm{visible}),
\quad\beta=\min(1,\sum_{j\,\mathrm{visible}}w_j),\quad
x=u+W^O_v[\beta r_1;\ldots;\beta r_{m/32}],\quad W^O_v\in\mathbb R^{d\times m}.
\]

Attention 的权重先验、全零分母处理、无写入边界导数服从原 KV/Attention 契约。
Full 只处理真实选中查询及矩形补齐，输出仍是 d 维：

\[
\operatorname{Full}(x)=x+W^{down}_v[
\operatorname{SiLU}(W^{gate}_v\operatorname{RMSNorm}(x))\odot
W^{up}_v\operatorname{RMSNorm}(x)].
\]

Gate/up 为 `f×d`，down 为 `d×f`。独立 KV 记忆参数（不含 FFN）为 `4*d*m+d*s`，
独立 Full FFN 为 `3*d*f`。缩窄 m 减少缓存 payload 和矩阵参数，不能直接推断峰值或吞吐：
历史拼接、整理、packing、Attention 临时量和其他模块的存活区间仍需实测。

## 仅记忆共享的归属

`memory_sharing=None` 沿用原 sharing；`region_shared` 只覆盖记忆模块。
主比较为 `sharing=independent, memory_sharing=region_shared, adapter_rank=8`。
FFN 虽注册在 Python memory 对象内部，其数学归属始终沿用 sharing。
对每种记忆矩阵用途 W：

\[
W_v=W^0_{g(v),W}+L^W_vR^W_v,\qquad W^0\in\mathbb R^{a\times b},\quad
L_v\in\mathbb R^{a\times r},\quad R_v\in\mathbb R^{r\times b},\quad r=\min(8,a,b).
\]

g 保留区域、角色、记忆种类和形状，不同用途不共享。KV 写入沿用原例外：同一 mixed
区内相容 K 节点共享区域 K/V 基座；其余矩阵仍区分角色。每个 KV 数学查询头分别适配，
不能把打包矩阵当成一份适配器。单节点组也保留基座＋适配器。

| 模块 | 首轮归属 |
|---|---|
| KV K/V、summary、逐头 query、output | 区域基座＋节点私有适配器 |
| SSD input（含 gate、卷积输入、dt 投影）、summary、read、output | 区域基座＋节点私有适配器 |
| SSD a_log、dt_bias、skip、rho | 按记忆区域组完全共享的小向量，无适配器 |
| SSD conv_kernel、conv_bias | 按记忆区域组完全共享；四抽头作为向量，无矩阵适配器 |
| Full FFN、接收、来源嵌入、描述、评分、区域网络、入口 | 保持原 sharing 归属 |
| KV、SSD 矩阵、摘要、卷积缓存、时间和 pending | 每节点、每独立样本私有 |

G 个 a×b 矩阵从 `G*a*b` 变为 `a*b+G*r*(a+b)`。共享基座梯度汇总全部实际使用
节点/样本/位置的贡献；适配器及状态仍私有。SSD 原始受约束参数先共享，再做原
exp/softplus 等映射，不在有效衰减上另加参数。本轮只选择这一共享变体。

## 执行和比较边界

同一共享基座的相容行可执行 `reshape(X,[-1,b]) @ W0.T`，另用节点 BMM 算私有适配器。
跨区域合批而基座不同，当前路径会按节点收集/重复展开基座做 BMM；减少存储不保证减少
复制或运算。记录实际矩阵形状、重复基座行、适配器与选择/执行/padding 行。

KV prepare 对接收候选做投影、历史拼接、摘要和窗口整理，Full 才执行查询、Attention、FFN。
SSD prepare 也在 Full 分配前，包含卷积、矩阵递推、摘要与描述读取；改为 SSD 不意味着
只为 Full 节点准备状态。独立 FFN 的动态选择 padding 仍会执行额外矩阵。
正常计时与逐阶段同步、设备 trace 等侵入式诊断分开。

固定图/d/s/f 与额外调整 FFN 的近总参数比较分表。同 B 与各自预算内最佳已测 B 分开；
有限阶梯不证明最大容量或全局最优。保留共同 Transformer 辅助参考：32层、d4352、
FFN15208、68个64维头、window128；它不替代 [A–D](benchmark-contract.md)，旧证据不改身份。
首输出指主干输出；prefill、稳定 decode、独立 batch 吞吐、完整训练分别报告。
工程性能和机制验证不能提供下游任务质量结论。

## 入口

选项均通过 ModelConfig.to_dict/from_dict 和 checkpoint 保存。
`configs/memory-candidates` 保存 small/medium/9b 及 near-total 显式配置。
训练入口新增 `--memory X_reverse --kv-width ... --ffn-width ... --memory-sharing region_shared`。
有界资格入口是 `python -m scripts.check_memory_candidates --device cpu --precision fp32 --output NEW.json`。
推理使用 `python -m benchmarks.inference --config configs/memory-candidates/9b-kv128.json
--devices npu:0 --precision fp32 --batch 8 --prefill 128 --prefill-chunk 32 --decode 16
--warmup 2 --samples 5 --allocated-budget-gib 48 --reserved-budget-gib 56 --output NEW.json`。
正式长任务经 [执行契约](execution-flows.md) 冻结、准入并持久运行。
不同数学候选不做逐值相等断言；同一候选的不同执行路径才比较值/VJP、梯度与恢复更新。
