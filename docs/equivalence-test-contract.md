# SettleGraph 等价性测试契约

> 本文规定怎样构造、执行和判定 SettleGraph 等价性测试，不是新的模型语义来源。
>
> 模型含义以 [实验语义、命名与数学符号](experiment-semantics-and-naming.md) 为准；执行路径和实施顺序见 [SettleGraph 实现与等价性验证计划](settlegraph-implementation-plan.md)。本文中的 formula ID 只标识测试 fixture 使用的确切公式，不把它们规定为科学实验的默认配置。
>
> 本文描述完整验收目标，不把仓库中已有的 eager reference、region-major reference 或定向单元测试自动算作资格通过。当前实现边界见实现计划第 1.1 节；只有本文要求的 fixture、comparator 和可追溯 artifact 全部存在并通过，才能形成对应 capability cell 的证据。

## 1. 声明范围与通过含义

一次等价性声明必须明确以下六个坐标：

1. logical Plan hash 与 concrete execution binding 的 typed Plan hash；
2. executor 及局部算子实现变体；
3. backend、精确设备或 host architecture 与 dtype；
4. forward、backward、optimizer、checkpoint 或端到端接入中的哪些方向；
5. fixture 集与本契约版本；
6. 使用的 comparator、容差和证据 artifact。

“executor 等价”表示同一 backend 和 concrete execution binding 下，两个 executor 对同一 fixture 通过本文要求的全部离散、浮点、状态、trace 和梯度比较。“跨 backend parity”表示 CPU 与 accelerator 分别在新进程中读取同一 CPU fixture，并在声明的容差内通过；它不表示浮点 bitwise 相同，也不表示两种 backend 的训练轨迹完全相同。

本文的核心范围是语义文档第 2.4 节定义的标准 Plan。共享可变状态、由模型 Tensor 产生 active budget 或公式未完整声明的自定义操作不能进入通过集合。实现可以拒绝超出范围的 Plan，但必须在运行前明确失败，不能静默改变语义。

## 2. Fixture bundle

一个 fixture bundle 是一次测试的不可变、自包含输入。它不依赖生成 fixture 时的随机进程继续存在。bundle 至少包含以下内容：

| 类别 | 必需内容 |
| --- | --- |
| 身份 | fixture schema 版本、fixture ID、内容 hash、生成命令或人工来源 |
| Plan | canonicalizer ID、规范化 logical/typed Plan bytes、logical Plan hash、concrete execution binding、typed Plan hash |
| 数值输入 | 在 CPU 上保存的 hidden、参数、可学习首状态和初始可变状态，保留原 dtype |
| 序列输入 | `sequence_id`、`token_position`、图执行/context mask、LM target mask、routing-stat mask |
| 控制输入 | Plan 开放时的 `requested_k`、reset 集合、chunk 切分、detach 边界和固定随机键 |
| 期望 | 成功或失败类别；成功时可含解析 golden、期望 route、充分统计和最终状态 |
| 梯度 | 标量目标定义、固定 cotangents、必查输入/参数/状态键及路径断言 |
| 路由分类 | exact tie、margin-safe、near-boundary 或 all-active |
| 端到端 Base 输入（适用时） | base/tokenizer 身份、输入与目标 Token、shift 规则、causal mask、position IDs，以及 decode case 的初始 KV cache |

所有 Tensor 都以 CPU artifact 作为交换表示。bundle 不能保存未解析 device 对象、进程内指针、绝对私有数据路径或依赖某个 executor 排列的状态 buffer。Attention 窗口按语义文档附录 A.5 规范化为按全局 Token 位置排序的有效三元组序列；无效 ring-buffer 槽位不进入 fixture。

跨 dtype 测试使用同一个 fixture family。family 保存可精确表示的逻辑源值或 CPU FP64 source，以及每种 concrete execution binding 的确定性 materialization 规则；每个 materialized bundle 分别记录 typed Plan hash 和 Tensor artifact hash。CPU 与 NPU FP32 必须读取 byte-identical 的 CPU FP32 bundle。FP64 对 FP32 的比较把两侧结果提升到 CPU float64 后进行，但不把两种 materialization 误称为同一个 typed Plan。

fixture loader 在构造模型或写入状态仓库前完成以下检查：schema、artifact hash、logical/typed Plan hash、Tensor key/shape/dtype/stride/storage group、mask 子集关系、唯一 `sequence_id`、连续位置、`requested_k` 容器的 region 键/shape/整数表示，以及状态所有者唯一性。状态所有者检查必须识别同一 backing storage 的不同 Tensor views，包括不重叠 views；只比较 object identity 不足以发现非法 alias。Tensor manifest 必须把规范 stride、storage offset、与物理地址无关的 storage group，以及包含 stride 间 holes 的完整 backing-storage bytes hash 纳入认证内容，使解除或新增 alias 关系、改变不可见 storage bytes 也会改变工件身份。`requested_k` 的值域是动态事件检查：只有候选非空的 region 事件才在 selector Read/Score 前解析该位置的值并校验 \([1,K^{\max}]\)；候选为空时不读取、也不对该位置的数值做范围判定。期望失败的 fixture 还保存规范化错误类别；loader 若走完相应入口仍未触发声明的缺陷，必须拒绝这个假负例。测试比较类别，不依赖任意一段异常文本。

资格 fixture 的 `sequence_id` 使用语义文档第 2.1 节稳定 ID 的字符串合法性和升序规则；它不是 logical Plan ID，也不进入 Plan hash。logical/typed Plan bytes 使用实现计划第 2.2 节声明的 canonicalizer。loader 必须先验证 canonicalizer ID、bytes 的规范性和 SHA-256，再解析 Plan；没有通过该 canonicalizer byte golden 的 executor 只能消费 bundle 中的规范 bytes，不能以本地 JSON 重写后得到的另一 hash 代替。

独立 SettleGraph fixture 没有语言模型 logits 或目标 Token，因此必须令第 5 节的 \(\alpha_{\mathrm{LM}}=0\)；它可以携带 LM target mask 以验证 mask 契约，但不能据此构造 LM loss。只有端到端 Base fixture 提供上表最后一行的输入并固定 logits-to-target 对齐后，才能令 \(\alpha_{\mathrm{LM}}\ne0\)。

### 2.1 首轮标准测试公式

为避免多个 executor 共同猜测“linear”“MLP”或“learned aggregate”的含义，首轮 fixture 使用下面的版本化公式。令

$$
\operatorname{Aff}(W,b,x)=Wx+b,
\qquad
\operatorname{SiLU}(x)=x\odot\sigma(x).
$$

除下文明确写出无 bias 的 RMSNorm 外，所有使用 \(\operatorname{Aff}\) 的 `TEST-*` 公式都要求 bias Tensor 作为稳定参数键存在。需要数值上无 bias 时把它显式置零；删除该参数或用 `bias=false` 改变参数 schema 属于另一公式，不得沿用原 formula ID。

`TEST-RMSNORM-V1` 对最后一维大小 \(d\) 的输入定义为

$$
\operatorname{RMSNorm}_{w,\epsilon}(x)
=
w\odot
\frac{x}{\sqrt{d^{-1}\sum_{i=1}^{d}x_i^2+\epsilon}},
$$

其中没有 bias；正数 \(\epsilon\) 作为公式常量写入 logical Plan，\(w\) 的稳定参数键/schema 写入 Plan 而 Tensor 数值由 fixture 的数值输入携带。\(w=1\) 不把 RMSNorm 变成 identity。若未来资格扩展需要无归一化操作，须先把定义为 \(N(x)=x\) 的独立公式 `TEST-NORM-IDENTITY-V1` 加入版本化 registry；它尚未注册，不能用于当前 `core-v1` fixture。

公式 ID、稳定参数键、参数 shape 和 dtype role 写入 logical Plan；参数 Tensor 的数值由 fixture bundle 单独携带，不进入 logical/typed Plan hash。每个版本化公式的必需键、允许键、默认值与参数 schema 也是该公式契约的一部分；未知键、缺失的必需键或改变数学/参数 schema 的值必须在执行前拒绝，不得被 executor 忽略。规范化还必须物化所有默认值；仅因“省略某键”与“显式写出同一默认值”而不同的两份原始配置，必须得到 byte-identical 的规范记录和同一 logical Plan hash。eager、packed、compiled 和 kernel 选择等实现变体字段另存于 execution binding/manifest，不混入公式语义配置。未来替换激活、bias 或归约规则必须使用新的 formula ID。

下面两个依赖父 edge ID 的 Aggregate 公式只作用于非入口 receiver。入口 receiver 的消息序列只有图边界 hidden，测试公式规定它原样返回；实现不得为此虚构语义父边或父边参数。

#### `TEST-AGG-EDGE-SOFTMAX-V1`

对按 edge ID 排列的非空消息 \((e_k,y_k)_{k=1}^n\)，每条固定父边有可训练标量 \(\eta_{v,e}\)：

$$
\alpha_k
=
\frac{\exp(\eta_{v,e_k})}
{\sum_{i=1}^n\exp(\eta_{v,e_i})},
\qquad
\operatorname{Aggregate}_v(\mathcal M)
=\sum_{k=1}^n\alpha_k y_k.
$$

softmax 只在实际到达的消息上归一化；`CLOSED` 边不占分母。

#### `TEST-AGG-EDGE-AFFINE-MEAN-V1`

每条固定父边有 \(W_{v,e}\in\mathbb R^{d_{\mathrm{model}}\times d_{\mathrm{model}}}\) 和 \(b_{v,e}\in\mathbb R^{d_{\mathrm{model}}}\)：

$$
\operatorname{Aggregate}_v(\mathcal M)
=
\frac1n\sum_{k=1}^n
\operatorname{Aff}(W_{v,e_k},b_{v,e_k},y_k).
$$

fixture 若需要 identity aggregate，显式取 \(W_{v,e}=I,b_{v,e}=0\)。

#### `TEST-AGG-TERMINAL-SOFTMAX-V1`

这个公式只用于图输出的 \(\operatorname{Aggregate}_{\mathrm{out}}\)。对按稳定 node ID 排列的非空实际终端消息 \((v_k,\widehat g_{v_k,t})_{k=1}^n\)，每个固定终端 receiver 有可训练标量 \(\eta^{\mathrm{out}}_v\)：

$$
\alpha_k^{\mathrm{out}}
=
\frac{\exp(\eta^{\mathrm{out}}_{v_k})}
{\sum_{i=1}^n\exp(\eta^{\mathrm{out}}_{v_i})},
\qquad
\operatorname{Aggregate}_{\mathrm{out}}
=
\sum_{k=1}^n
\alpha_k^{\mathrm{out}}\widehat g_{v_k,t}.
$$

softmax 只在本次实际 active 的终端消息上归一化；未 reached 或 inactive 的固定终端不占分母。参数键仍为全部固定终端按 node ID 定义，即使某个终端在特定 fixture 中从未 active，也不得因 executor 的实际消息布局而改变。

#### `TEST-READ-PROJ-V1`

selector readout 的输入 \(x_{v,t}^{\mathrm{sel}}\) 由时序决定：content 使用 \(m_{v,t}\)，pre 使用串接的 \([m_{v,t};s^-_{v,t}]\)，post 使用 \([m_{v,t};\widetilde s_{v,t}]\)。状态含多个分量时按 Plan 中的稳定分量 ID 展平并串接。读出为

$$
r^{\mathrm{sel}}_{v,t}
=
\operatorname{Aff}(W^{\mathrm{sel}}_v,b^{\mathrm{sel}}_v,
x_{v,t}^{\mathrm{sel}}).
$$

pre/post 使用 `TEST-READ-PROJ-V1` 时，进入串接的可微状态分量必须具有 Plan 已声明的固定 shape，因此投影输入维度固定。规范窗口 Attention 的有效长度会变化，不能把未规定 padding 的窗口直接 flatten 后沿用这个 formula ID；它使用下面的有界摘要公式。

#### `TEST-READ-STATE-RMS-SUMMARY-PROJ-V1`

这个读出把当前归一化输入 \(m_{v,t}\) 与状态的一个标量摘要串接。对固定 shape 的非空 Tensor 状态 \(s\)，令

$$
\chi(s)
=
\sqrt{
\frac{\lVert\operatorname{vec}(s)\rVert_2^2}
{\operatorname{numel}(s)}
}.
$$

对语义文档附录 A.5 的规范窗口 Attention 状态，若有效长度 \(n_s>0\)，令

$$
\chi(s)
=
\sqrt{
\frac{
\lVert\mathbf K(s)\rVert_F^2
+
\lVert\mathbf V(s)\rVert_F^2
}{
n_s(d_k+d_v)
}
}.
$$

无状态或空状态取 \(\chi(s)=0\)。Attention 的有效 Token 位置不进入数值摘要，无效物理 buffer 槽位也不进入；keys 和 values 按规范时间顺序参与上式。selector 时序决定这里的 \(s\) 是旧状态还是 proposal。最终读出为

$$
r^{\mathrm{sel}}_{v,t}
=
\operatorname{Aff}
\left(
W^{\mathrm{sel}}_v,
b^{\mathrm{sel}}_v,
[m_{v,t};\chi(s)]
\right),
$$

其中 \(W^{\mathrm{sel}}_v\) 的输入宽度为 \(d_{\mathrm{model}}+1\)。该公式没有额外 epsilon；在摘要输入恰为全零时，测试公式规定 \(\partial\chi/\partial s=0\)，从而固定这个不可微点的 VJP，非零点仍使用上式的通常导数。若改变摘要、加入位置/有效长度、零点次梯度或归约精度策略，必须使用新的 formula ID。

#### `read.selector.content-rms.v1`

这个无参数读出只使用当前归一化输入 \(m_{v,t}\)，返回 shape 为 \([1]\) 的向量

$$
r^{\mathrm{sel}}_{v,t}
=
\left[
\sqrt{
\frac{1}{d_{\mathrm{model}}}
\sum_{i=1}^{d_{\mathrm{model}}}m_{v,t,i}^{2}
}
\right].
$$

它不读取 receiver state 或 selector-history，也不增加 epsilon；零向量的结果为零，且测试公式规定该点对 \(m\) 的 VJP 为零。该公式与 `TEST-READ-STATE-RMS-SUMMARY-PROJ-V1` 中对状态的摘要不是同一操作，不能互换 formula ID。

#### `TEST-HISTORY-ACTIVE-EMA-V1`

这是只服务 selector 的 node-level history。令 \(c^-_{v,t}\) 为当前 Token 前的标量，选择完成后按

$$
c_{v,t}
=
\begin{cases}
\lambda_h c^-_{v,t}
+(1-\lambda_h)\mathbf 1[v\in\mathcal A_{\mathcal R,t}],
&v\in\mathcal C_{\mathcal R,t},\\
c^-_{v,t},&v\notin\mathcal C_{\mathcal R,t},
\end{cases}
$$

写回，其中 \(0\le\lambda_h<1\)，首值和 \(\lambda_h\) 写入 Plan。写回 stop-gradient，并从同一稳定序列的下一执行 Token 起可见；使用该 history 的 pre selector 把 \(c^-_{v,t}\) 作为 `TEST-READ-PROJ-V1` 输入的最后一个分量。它不属于 receiver Observe set。

这个 formula ID 闭合的是一个 node-level fixture 的数学与时序，不同时规定所有 selector-history 变体共用的 Plan owner/字段/序列化 schema。后者在实现计划第 1.1 节所列的通用 schema 闭合前仍可被实现明确拒绝；该拒绝表示相应 capability 尚未实现，不能把本公式从完整资格目标中删除。

#### `TEST-SCORE-CONST-V1`、`TEST-SCORE-LINEAR-V1` 与 `TEST-SCORE-MLP-V1`

固定分数 fixture 直接读取 Plan 中每个 node 的有限标量 \(c_v\)：

$$
a_{v,t}=c_v.
$$

线性和两层 MLP 分数先按稳定顺序构造

$$
x_{v,t}^{\mathrm{score}}
=
[r^{\mathrm{sel}}_{v,t};c^{\mathrm{ctx}}_{\mathcal R,t}],
$$

公共摘要为空时只保留 readout。然后分别计算

$$
a_{v,t}
=
(w_v^{\mathrm{score}})^\top x_{v,t}^{\mathrm{score}}
+b_v^{\mathrm{score}},
$$

或

$$
z_{v,t}
=
\operatorname{SiLU}
\left(
\operatorname{Aff}(W_{1,v},b_{1,v},x_{v,t}^{\mathrm{score}})
\right),
\qquad
a_{v,t}=w_{2,v}^{\top}z_{v,t}+b_{2,v}.
$$

这些 Score 参数默认按稳定 node ID \(v\) 独立；fixture 若验证共享只读参数，必须用 Plan 中显式参数组让相应 nodes 引用同一个 Tensor，并按同一参数键比较累积梯度。

#### `score.constant.v1`、`score.fixed-by-node.v1` 与 `score.read-sum.v1`

`score.constant.v1` 把 Plan 中的一个有限标量 \(c\) 用于本次所有 candidates：

$$
a_{v,t}=c.
$$

其规范配置的 `value` 为该标量，省略时默认为 \(0\)；规范化 Plan 必须物化 `value`，并拒绝 bool、NaN 或无穷值。

`score.fixed-by-node.v1` 与 `TEST-SCORE-CONST-V1` 的按 node 固定分数公式相同：

$$
a_{v,t}=c_v.
$$

其规范配置使用 `values_by_node`，键集必须精确等于该 region 的全部固定 node IDs，每个值是有限标量，且没有默认值。两个同义 formula ID 在 fixture 和证据中仍保留原值，不得为了实现方便静默改写 ID。

`score.read-sum.v1` 没有可训练参数。令

$$
x_{v,t}^{\mathrm{score}}
=
[r^{\mathrm{sel}}_{v,t};c^{\mathrm{ctx}}_{\mathcal R,t}],
$$

其分数是

$$
a_{v,t}
=
\sum_i x_{v,t,i}^{\mathrm{score}}.
$$

首轮 reference 只实现 `context.none.v1`，因此该 binding 下的公共摘要为空、分数即 readout 各分量之和。不得用一个只写入 Score 配置而没有对应 selector-context 公式和输入的非零 `context_dim` 扩张该语义。

#### `TEST-NODE-AFFINE-V1` 与 `TEST-NODE-SWIGLU-V1`

解析小图使用

$$
g_{v,t}
=
h_{v,t}
+\operatorname{Aff}(W_v^{\mathrm{node}},b_v^{\mathrm{node}},m_{v,t}).
$$

SwiGLU 小图使用

$$
x_{v,t}=N_{F,v}(u^{\mathrm{node}}_{v,t}),
$$

$$
E_v(x)
=
\operatorname{Aff}
\left(
W_{o,v},b_{o,v},
\operatorname{SiLU}(\operatorname{Aff}(W_{g,v},b_{g,v},x))
\odot
\operatorname{Aff}(W_{u,v},b_{u,v},x)
\right),
$$

并代入语义文档第 2.2 节的双 residual。所有 bias 都存在；fixture 需要无 bias 时把对应 Tensor 显式置零。取状态读出为零且 \(W_{o,v}=0,b_{o,v}=0\) 可构造 identity forward，但 \(W_{o,v}\) 在该点的梯度一般不为零，因此不能据此声称完整训练梯度与 Base 模型等价。

EMA、Gated DeltaNet、Attention、Emit 和 `BAL-AVAIL-SOFT` 直接使用语义文档中的公式。为了把其中原本只写作“向量归一化”的 \(N_k,N_q\) 也闭合，首轮 Gated DeltaNet 和窗口 Attention fixture 统一使用

$$
\operatorname{L2Norm}_{\epsilon}(x)
=
\frac{x}{\max(\lVert x\rVert_2,\epsilon)},
\qquad \epsilon>0,
$$

并在 logical Plan 的对应 Update 配置中保存 \(\epsilon\)。这里不在平方根内额外加 epsilon；零向量的结果为零。连续 VJP fixture 要求 \(\lVert x\rVert_2\ne\epsilon\)；精确相等边界只做 forward 与 finite 检查，本契约不要求不同 autograd 实现在该非光滑点选取相同导数。

直接复用语义公式的首轮序列化绑定如下。表中一个 ID 只指向右侧的确切计算；改变归一化、bias、门控、缩放或时序时必须换新 ID。

| Plan 角色与 formula ID | 确切绑定 |
| --- | --- |
| Aggregate `agg.mean.v1` | 语义文档第 2.1 节中对实际消息的算术平均；同一 ID 可用于图输出聚合 |
| normalization `norm.rms.v1` | 与 `TEST-RMSNORM-V1` 完全相同，包括无 bias、可学习 \(w\) 和 Plan 中的正 epsilon |
| Update `update.none.v1` | 无 receiver state、proposal 或 commit |
| Update `state.ema.v1` / FFN Read `read.ffn.ema.v1` | 语义文档附录 A.3 的 EMA Update 与 \(W^{\mathrm{out}}s\) 读出；首轮绑定的 \(\lambda\) 是 logical Plan 记录的固定有限标量且 \(0\le\lambda<1\)，逐维或可学习衰减须另建 formula ID 并定义参数化 |
| Update `state.gdn.v1` / FFN Read `read.ffn.gdn.v1` | 语义文档附录 A.4，且 \(N_k=N_q=\operatorname{L2Norm}_{\epsilon}\) |
| Update `state.attention-window.v1` / FFN Read `read.ffn.attention-window.v1` | 语义文档附录 A.5 的 AppendEvict 和缩放点积 Attention，且 \(N_k=N_q=\operatorname{L2Norm}_{\epsilon}\) |
| selector Read `read.selector.content.v1` | \(r^{\mathrm{sel}}=m\) |
| FFN Read `read.ffn.zero.v1` | 返回 \(d_{\mathrm{model}}\) 维零向量 |
| NodeCompute `node.identity.v1` | \(g=h\)，不读取 \(m\) 或 receiver state |
| Emit `emit.hard.v1`、`emit.hst.v1`、`emit.softp.v1` | 分别精确对应语义文档第 2.3 节的 `EMIT-HARD`、`EMIT-HST`、`EMIT-SOFTP` |
| selector context `context.none.v1` | \(c^{\mathrm{ctx}}\) 是空向量，Score 输入不追加公共摘要 |
| selector history `history.none.v1` | 不存在 selector-history 状态、读出或写回 |
| active budget `k.fixed.v1` / `k.input.v1` | 分别使用 Plan 固定整数或候选非空事件的运行期 `requested_k`，值域与时序遵守语义文档第 2.3 节 |

首轮 fixture 的 receiver input/FFN normalization 使用数学上完全同义的 `TEST-RMSNORM-V1` 或 `norm.rms.v1`。报告仍保留 fixture 实际使用的 ID，不在证据中静默改写。当前 registry 没有无归一化公式；未来若注册 `TEST-NORM-IDENTITY-V1` 或其他 normalization，必须记录完整公式和适用常量，只写“identity”“RMSNorm”或“LayerNorm”不足以生成 golden。

### 2.2 首轮 Plan formula config schema

本节固定 logical Plan schema v1 中上述 reference 公式的规范配置。每项配置都含下表的规范 `type`、一个本节绑定的 `formula_id` 和“其他规范键”列出的全部键；该列没有列出的键不得出现。原始配置的 `type` 可做小写化和连字符转下划线的纯语法规范化；规范记录只保存表中拼写，不保留等价 alias。

这张表只闭合当前 eager reference 子集。`TEST-HISTORY-ACTIVE-EMA-V1` 的局部数值公式已在上文闭合，但在通用 history owner/schema 定稿前不进入本表；跨 node/site 参数组和尚未注册的 `TEST-NORM-IDENTITY-V1` 同理。当前 reference 必须在运行前拒绝这些未闭合能力，它们仍是第 7 节完整资格目标中的未完成项。

表中的“派生”值从同一 logical Plan 的已声明 shape 或 region 常量决定，必须物化到规范配置并与来源一致。输入省略有默认值的键与显式写出该值必须产生 byte-identical 记录。`required` 表示没有默认，不能从其他字段猜测。

| Plan 字段 | 规范 `type` 与 formula ID | 其他规范键 |
| --- | --- | --- |
| input/FFN normalization | `rmsnorm`：`norm.rms.v1` 或 `TEST-RMSNORM-V1` | `eps`，默认 \(10^{-6}\) |
| receiver Aggregate | `mean`：`agg.mean.v1` | `output_shape`，派生为 \([d_{\mathrm{model}}]\) |
| receiver Aggregate | `edge_softmax`：`TEST-AGG-EDGE-SOFTMAX-V1` | `output_shape`，派生为 \([d_{\mathrm{model}}]\) |
| receiver Aggregate | `edge_linear_mean`：`TEST-AGG-EDGE-AFFINE-MEAN-V1` | `bias=true`；`output_shape` 派生为 \([d_{\mathrm{model}}]\) |
| Update | `none`：`update.none.v1` | `state_shape=[]` |
| Update | `ema`：`state.ema.v1` | `state_dim`，默认为声明的 \(d_s\)；`decay`，默认 \(0.9\)；`learnable_decay=false`；`state_shape`派生为 \([d_s]\) |
| Update | `gdn`：`state.gdn.v1` | `key_dim` 与 `value_dim` 均 required；`norm_eps` 默认 \(10^{-12}\)；`state_shape`派生为 \([d_k,d_v]\) |
| Update | `attention_window`：`state.attention-window.v1` | `key_dim`、`value_dim`、`window` 均 required；`norm_eps` 默认 \(10^{-12}\)；`state_shape`派生为 \([W,d_k,d_v]\) |
| selector Read | `content`：`read.selector.content.v1` | 只用于 content 时序；`out_dim`派生为 \(d_{\mathrm{model}}\)；`output_shape`派生为 \([d_{\mathrm{model}}]\) |
| selector Read | `content_norm`：`read.selector.content-rms.v1` | 只用于 content 时序；`out_dim=1`；`output_shape=[1]` |
| selector Read | `content_linear`：`TEST-READ-PROJ-V1` | 只用于 content 时序；`out_dim`派生为 selector readout 维度 \(d_r\)；`output_shape`派生为 \([d_r]\) |
| selector Read | `content_state_linear`：`TEST-READ-PROJ-V1` | 只用于 pre/post 时序和固定 shape 的非空 Tensor state；`out_dim`派生为 \(d_r\)；`output_shape`派生为 \([d_r]\) |
| selector Read | `content_state_summary_linear`：`TEST-READ-STATE-RMS-SUMMARY-PROJ-V1` | 只用于 pre/post 时序；`out_dim`派生为 \(d_r\)；`output_shape`派生为 \([d_r]\) |
| FFN Read | `zero`：`read.ffn.zero.v1` | `output_shape`派生为 \([d_{\mathrm{model}}]\) |
| FFN Read | `state_default`：与 Update 一致的 `read.ffn.zero.v1`、`read.ffn.ema.v1`、`read.ffn.gdn.v1` 或 `read.ffn.attention-window.v1` | `output_shape`派生为 \([d_{\mathrm{model}}]\) |
| NodeCompute | `identity`：`node.identity.v1` | `output_shape`派生为 \([d_{\mathrm{model}}]\) |
| NodeCompute | `affine_residual`：`TEST-NODE-AFFINE-V1` | `bias=true`；`output_shape`派生为 \([d_{\mathrm{model}}]\) |
| NodeCompute | `double_residual_swiglu`：`TEST-NODE-SWIGLU-V1` | `hidden_dim`，默认 \(4d_{\mathrm{model}}\)；`bias=true`；`output_shape`派生为 \([d_{\mathrm{model}}]\) |
| Emit | `hard`：`emit.hard.v1` | `output_shape`，派生为 \([d_{\mathrm{model}}]\) |
| Emit | `hst`：`emit.hst.v1` | `zeta`，默认 \(1\)；`output_shape`派生为 \([d_{\mathrm{model}}]\) |
| Emit | `softp`：`emit.softp.v1` | `output_shape`，派生为 \([d_{\mathrm{model}}]\) |
| Score | `constant`：`score.constant.v1` | `value`，默认 \(0\) |
| Score | `fixed`：`score.fixed-by-node.v1` 或 `TEST-SCORE-CONST-V1` | `values_by_node` required |
| Score | `linear`：`TEST-SCORE-LINEAR-V1` | `bias=true`；`shared_parameters=false`；`context_dim=0` |
| Score | `mlp`：`TEST-SCORE-MLP-V1` | `hidden_dim`，默认 \(\max(4,d_r)\)；`bias=true`；`shared_parameters=false`；`context_dim=0` |
| Score | `read_sum`：`score.read-sum.v1` | `context_dim=0` |
| selector context/history | `none`：`context.none.v1` / `history.none.v1` | 无 |
| active budget | `fixed`：`k.fixed.v1` | `value` required |
| active budget | `input`：`k.input.v1` | `field="requested_k"`、`minimum=1`，以及 `maximum` \(=K^{\max}_{\mathcal R}\)，三者均 required |
| output Aggregate | `mean`：`agg.mean.v1` | `output_shape`，派生为 \([d_{\mathrm{model}}]\) |
| output Aggregate | `node_softmax`：`TEST-AGG-TERMINAL-SOFTMAX-V1` | `output_shape`，派生为 \([d_{\mathrm{model}}]\) |

`bias=true`、`learnable_decay=false`、`shared_parameters=false` 和 `context_dim=0` 都是该 formula ID 的固定 schema 值，不是可切换开关；写出其他值必须拒绝。所有 dimension 是非 bool 正整数，所有标量常数是非 bool 有限数；规范化把数学上相同的整数/浮点输入（如 `zeta=1` 与 `zeta=1.0`）统一为同一标量数值表示，并把负零规范为正零。作为公式实数输入的原始整数还必须位于 JSON/IEEE-754 safe-integer 区间 \([-(2^{53}-1),2^{53}-1]\)；超出该区间必须拒绝，不能先转 binary64、静默舍入后与另一个整数产生相同 hash。`eps` 和 `norm_eps` 严格为正，`decay` 满足 \(0\le\texttt{decay}<1\)。`values_by_node` 的键集精确等于 region 的固定 node IDs，值均为非 bool 有限标量并使用相同规范数值表示。

selector readout 的首轮规范 shape 是一维 \([d_r]\)；多维声明不能只取首维执行，必须在 Plan 验证时拒绝。Update 的 `state_shape` 必须分别与 EMA、Gated DeltaNet 和窗口 Attention 表中的派生值一致；它不能作为被 executor 忽略的第二份状态声明。其中窗口 Attention 的 \([W,d_k,d_v]\) 是 logical Plan v1 对复合 Attention 状态的尺寸描述符，不表示一个 shape 为 \([W,d_k,d_v]\) 的物理 Tensor；它的规范状态仍是语义文档附录 A.5 定义的有效位置、keys 和 values 有序序列。

selector timing 与 Read 类型也在 Plan 阶段交叉校验。content 时序只允许不读取 state 的 `content`、`content_norm` 或 `content_linear`；pre/post 时序必须使用显式带 state 输入的 `content_state_linear` 或 `content_state_summary_linear`。不能让 `content_linear` 在 pre/post 中静默丢弃可见状态，也不能让带 state 的投影在 content 时序收到一个临时的 absent 值。这里的 post 仍受语义文档“SD 不使用 post-update 选择”的上层约束。

完整 fixture/checkpoint qualification 还要求一个版本化、实现无关的 parameter-schema manifest。manifest 对每个公式参数保存由 site、field、稳定 node/region/edge ID 和公式内参数角色组成的 logical key，以及 formula ID、shape、dtype role 和可选只读参数组 ID；条目按 logical key 规范排序。公式内角色就是第 2.1 节中的 (w,W,b,\eta,\beta) 等已定义量，不能换成某个 eager module 的属性路径。executor 的 `state_dict` 名称可以作为该实现的装载 locator，但不能充当跨 executor 的参数身份；每个 executor 必须显式证明 locator 与 logical key 一一对应。参数 Tensor 数值仍只由 bundle/checkpoint 携带，不进入 Plan hash。

当前 eager reference 已能对单个 SettleGraph site 从 Plan 派生 `tide.parameter-schema.v1`：逻辑记录包含 field、稳定 node/region/edge/terminal ID、公式参数角色、formula ID、shape 和 dtype role，eager locator 位于独立 binding，并校验与 `named_parameters()` 一一对应。跨 sites 的 site ID、通用参数组及共享兼容性 schema 仍未闭合；当前序列化 bundle 的正向 round trip 也没有达到第 7 节的语料、独立 golden 和数量门槛。因此 parameter schema 的实现不等于 fixture、checkpoint 或 capability qualification 已通过。

### 2.3 失败类别 v1

期望失败的 fixture 保存 `error_schema="tide.failure.v1"`、一个 `phase` 和非空的 `codes` 数组；不比较 Python/C++ 异常类名、堆栈或任意错误文本。`codes` 去重后按稳定字符串顺序排列。每个动态失败 fixture 必须只注入一个可到达的独立缺陷；静态 Plan mutant 也应只做一次有名变换。若这一次变换按规范必然同时违反多项静态不变量，则保存 validator 应报告的完整 code 集合，不能由 executor 任意选择其中一条。

首轮允许的 `(phase, code)` 如下：

| `phase` | `code` | 规范边界 |
| --- | --- | --- |
| `artifact` | `artifact.integrity` | bundle/checkpoint 内容 hash、Plan canonical bytes 或 weights-only-safe 解码失败 |
| `artifact` | `artifact.schema` | hash 正确且可安全解码，但 bundle 根版本、键集、expected envelope、Tensor manifest/key/shape/dtype 等 artifact 结构不合法 |
| `plan` | `plan.schema` | Plan 根结构、JSON 类型、schema version 或稳定 ID 编码不合法 |
| `plan` | `plan.topology` | ID 唯一性、边/region 归属、DAG、边界路径、HB 或静态容量约束失败 |
| `plan` | `plan.formula` | 已注册公式的键集、默认/固定值、shape、timing、profile 或 K 声明不合法 |
| `binding` | `binding.invalid` | dtype role 缺失、冲突或 concrete binding 记录不合法 |
| `capability` | `capability.unsupported` | Plan/binding 在语义上合法，但所请求 executor、公式或 backend cell 未实现 |
| `input` | `input.schema` | 调用容器、Tensor shape/dtype、重复 `sequence_id` 或 reset 集合不合法 |
| `input` | `input.mask` | mask shape/dtype 或子集关系不合法 |
| `input` | `input.position` | 新序列起点、连续性、重放、倒序或跳号不合法 |
| `event` | `input.requested_k` | 候选非空事件读取到缺失、非整数或越界的运行期 K；候选为空不能产生此 code |
| `state` | `state.schema` | 状态键、shape、dtype、Attention 有效窗口或下一位置不合法 |
| `state` | `state.owner_alias` | owner 不唯一，或不同 owner 共享同一 backing storage |
| `execution` | `execution.local_operation` | 已通过静态/输入校验的声明公式在事件执行时明确失败 |
| `execution` | `execution.empty_terminal` | 标准 Plan 的执行 Token 没有终端消息 |
| `checkpoint` | `checkpoint.integrity` | checkpoint 内容 hash、安全解码或发布完整性失败 |
| `checkpoint` | `checkpoint.schema` | checkpoint root/key/type/version 结构不合法 |
| `checkpoint` | `checkpoint.compatibility` | Plan/binding/model/optimizer/状态身份或 shape/dtype 不兼容 |
| `checkpoint` | `checkpoint.commit` | 注入式 load/commit 失败；必须同时验证公开状态回滚 |
| `runtime` | `runtime.configuration` | device/index/dtype/seed 请求本身不合法 |
| `runtime` | `runtime.unavailable` | 请求合法但目标设备、插件、算子或精确能力不可用 |

已注册 formula ID 放到错误 field/type、已注册 type 缺失或使用未知 formula ID，以及已注册公式的键、固定值或派生 shape 不合法，均属于 `plan.formula`。只有 Plan 自带完整数学定义、通过 generic Plan 语义校验，而指定 executor 尚未实现的自定义公式，才属于 `capability.unsupported`；一个没有自包含公式定义的未知 ID 不能借此绕过 Plan gate。

阶段优先级按入口分别固定：

- 普通 fixture/executor：artifact integrity → artifact schema → Plan schema gate → Plan topology/formula → binding → capability → whole-call input/mask/control container → state/owner alias → position → reached event 的 K → local operation/执行不变量；
- checkpoint load：调用参数和目标对象预检 → artifact integrity → payload schema → compatibility → commit，前四步不得修改 model、optimizer、RNG 或公开 sequence state；
- runtime：configuration → availability/minimal capability probe → fixture 执行。

`phase` 取最早失败阶段，`codes` 只包含该阶段的 code，后续阶段不再求值。Plan schema 是 gate：根结构或类型不足以安全解释时只报告 `plan.schema`；schema 通过后，validator 才聚合同一 mutant 导致的独立 topology/formula codes。静态 codes 去重并按稳定字符串排序。动态资格 fixture 只有一个可到达故障，因此不得依赖 token-major、region-major 或并行执行器“谁先抛异常”的物理顺序。

细粒度 mutant 覆盖使用 fixture manifest 的 `mutation_kind` 统计，例如 cycle、duplicate ID、wrong-dispatch formula 或 aliasing views；首版 executor envelope 只比较上表的稳定粗粒度 code，不把异常文本或实现私有 leaf reason 当作等价条件。随机非法语料的覆盖报告按不同 fixture 计数，每个实际声称支持的 code 至少有一个单缺陷 fixture。当前 eager reference 已提供 `tide.failure.v1` envelope、allowlist、比较与异常捕获 helper；Plan validator 自产 schema/topology/formula code，能由异常类型唯一确定的 code 直接映射，其他跨多个规范阶段的粗异常仍须由已知调用阶段显式提供 code，禁止解析异常文本猜测。保存器还可从合法 payload 应用并认证三种命名 mutation，真实落盘 Plan topology、mask 和 state storage-alias 负 bundle，并拒绝只声明 failure 的合法 payload。这只闭合三个代表性 loader-stage 负例；在全部 code 的单缺陷 bundle、两个 executor 的动态对照、96-case 覆盖和失败收缩补齐前，仍不能计作完整 error-category qualification。

## 3. 独立 oracle 与 exact trace

逐 Token 解释器是调度 oracle，但不是局部公式的唯一 oracle。至少一组人工 fixture 必须从保存的有理数或可精确枚举的小 Tensor 直接写出期望值，且生成期望值的代码不能调用被测执行器共用的 Aggregate、Update、Read、Score、Top-K、NodeCompute、Emit、状态提交或 balance-loss helper。

每个成功的 exact-trace fixture 按以下顺序保存事件：

1. 调用输入、三个 masks、序列位置、reset 和调用前状态；
2. 每个 Token 的入口边界消息；
3. 对每个执行 Token，保存每条固定边的 `DATA`/`CLOSED`，`DATA` 时保存 payload；
4. 每个 receiver 的父消息序列、reached、聚合 hidden \(h\)、归一化输入 \(m\) 和 \(s^-\)；
5. proposal 是否存在及其值、selector readout 和公共摘要；
6. 每个 region 的 candidates、logits、probabilities、请求 K、实际 K 和 Top-K IDs；
7. Observe/active、\(s^{\mathrm{cmp}}\)、NodeCompute 的 \(g\) 与 Emit 的 \(\widehat g\)；
8. receiver state 与 selector-history 的 staged write；
9. 按 node ID 排列的终端消息、输出聚合和图输出；
10. 调用后规范状态、下一位置与 LM/balance 充分统计量。

不存在的 proposal、readout、payload 或梯度使用显式的 absent 标记，不能以空 Tensor、零 Tensor 或缺少字段三种方式混用。trace 的规范排序键为 site ID、`sequence_id`、全局 Token 位置、region 规范拓扑序及 region ID、node ID、edge ID；executor 的实际完成时刻和 packed row 不进入排序。

reached 的 forced-active singleton 不是 selector 的 absent 事件：它必须保存按声明时序得到的实际 readout、声明 Score 公式产生的实际单元素 logits、精确的单元素 probability \([1]\)、请求值/实际值 1，以及唯一 node 的 Top-K ID。active 成员关系不读取 logit 排名，但不得以合成的零 logit代替实际 Score。该 singleton 未 reached 时才是候选为空事件，不执行 Read/Score。

无执行位置必须精确记录旁路输出等于入口 hidden、没有 selector event、没有状态 staged write。期望失败的事务 fixture 记录失败前可用于诊断的私有 trace，但正式结果必须标为 failure，并证明公开状态 hash、下一位置和 artifact 集与调用前完全相同。

## 4. Comparator

### 4.1 离散与结构量

以下量要求 exact：schema、logical/typed Plan hash、Tensor key、shape、声明 dtype、mask、状态 owner、候选及其顺序、requested/effective K、reached/Observe/active/send、Top-K IDs、edge status、Attention 有效位置、错误类别和 checkpoint key 集。

exact 相等不能替代单边语义 invariant。每个成功 trace 还独立检查：每个执行 Token 的每条固定边恰好结算一次；每个 region 恰好结算一次；candidates 恰为 reached members 且 active 是其合法大小子集；Observe set 与 N/SD/BO 公式相同；普通 selector probability 非负并在当前 binding 的浮点门槛内和为 1；reached 的 forced-active singleton 有实际 Read/Score 值、probability 精确为 1 且唯一 candidate active；成功 Token 的终端消息非空；所有消息 hidden、状态 owner 和 Attention 有效长度满足 Plan；非执行位置没有图事件。任一 invariant 失败时，即使两个 executor 产生相同错误结果也不能通过。

同一 concrete execution binding 的输出 dtype 必须相同。跨 binding 比较时，每一侧必须符合自身声明的 dtype；device placement 由该进程的运行 manifest 和 profiler 证明，不能通过把 accelerator case 实际放到 CPU 后获得数值通过。

### 4.2 浮点量

先在产生结果的 backend 上检查有限性，再复制到 CPU 并在 float64 中计算误差。任一待比较浮点值出现 NaN、正无穷或负无穷，case 立即失败，即使参考与候选在相同位置出现相同非有限值也不接受。

当前公式/Plan schema 的资格门槛只适用于四个核心 dtype roles 同为 FP64 或同为 FP32 的 binding，归约使用输入 role 的 dtype。FP16/BF16 虽可作为 runtime capability probe 的请求候选，但在逐公式 accumulation role、舍入规则和低精度门槛进入 typed Plan 前，不属于本契约已闭合的 SettleGraph 资格 cell。

以 \(x\) 为 reference、\(y\) 为 candidate，逐元素要求

$$
|y_i-x_i|
\le
\operatorname{atol}
+\operatorname{rtol}|x_i|.
$$

小 fixture 的默认门槛为：

| 比较 | `atol` | `rtol` |
| --- | ---: | ---: |
| CPU FP64：解析/逐 Token/packed/特化 | \(10^{-10}\) | \(10^{-8}\) |
| 同一 backend FP32：executor 或 eager/optimized | \(10^{-6}\) | \(10^{-5}\) |
| CPU FP32 与 accelerator FP32 | \(10^{-4}\) | \(10^{-4}\) |

这些门槛适用于 hidden、状态、logits、probability、loss、充分统计和梯度。具体量可以在 fixture manifest 中使用更严门槛；放宽门槛会形成新的契约版本和能力声明，不能由测试运行临时决定。低精度、长序列、短训练或性能 workload 使用独立的数值策略，不能借其门槛把小 fixture 判为通过。

comparator 除 pass/fail 外保存最大绝对误差、最大相对误差、最坏元素的稳定路径、reference/candidate 值和容差。只报告一个全局最大值不足以定位状态或事件错误。

### 4.3 路由边界分类

Top-K 按语义文档第 2.3 节直接排序 logits；soft probability 用于 Emit、辅助项和 trace，不作为有限精度路由排序输入。对一个普通选择事件，令 \(C=|\mathcal C_{\mathcal R,t}|\) 为候选数，\(K=\min(K^{\mathrm{req}}_{\mathcal R,t},C)\) 为实际 active 数。对 \(C>K\) 的事件，把 logits 按“值降序、平票 node ID 升序”排列，令

$$
\Delta_K=a_{(K)}-a_{(K+1)}\ge0.
$$

使用当前 logits comparator 的门槛定义 guard band

$$
g_K
=
4\left[
\operatorname{atol}
+\operatorname{rtol}
\max(|a_{(K)}|,|a_{(K+1)}|)
\right].
$$

事件分类为：

- **exact tie**：fixture 解析构造 \(\Delta_K=0\)，用于验证稳定 node ID 平票；
- **margin-safe**：reference 中 \(\Delta_K>g_K\)，用于自然路由等价和梯度差分；
- **near-boundary**：\(0<\Delta_K\le g_K\)，或 fixture 显式构造容差量级的边界扰动；
- **all-active**：\(K\ge C\)，不存在第 K/K+1 边界。

所有声称端到端等价的 case 都要求离散 route exact，包括 near-boundary。near-boundary route 不一致应报告为 route-sensitive failure，不能用浮点 tolerance 忽略。随后可以用 reference route replay 再执行一次，把“selector 边界差异”和“给定同一路由后的 receiver/状态差异”分开；replay 通过是附加诊断，不改变自然路由 case 的失败状态。

## 5. 梯度与 optimizer 契约

每个 backward fixture 保存一个或多个固定标量目标。通用形式为

$$
J
=
\sum_{b,t}\langle q^{\mathrm{out}}_{b,t},b_{\mathcal G,b,t}\rangle
+\sum_k\langle q^{\mathrm{state}}_k,S^{\mathrm{final}}_k\rangle
+\alpha_{\mathrm{LM}}\mathcal L_{\mathrm{LM}}
+\alpha_{\mathrm{bal}}\mathcal L_{\mathrm{bal}}^{\mathrm{SG}},
$$

其中 cotangents \(q\) 和系数都保存在 fixture；未使用的项系数为零，独立 SettleGraph fixture 按第 2 节令 \(\alpha_{\mathrm{LM}}=0\)。状态项只包含 Plan 声明为可微的规范 Tensor，不把 Token 位置、ring head、active count 等离散元数据当作可微量。executor 必须对完全相同的 \(J\) 计算 VJP。

必查梯度键至少包含：

- 所有执行位置的入口 hidden；
- 声明可微的初始 receiver state；
- fixture 实际使用的 Aggregate、Update、selector Read、Score、状态 Read、NodeCompute 和 Emit 参数；
- Base 接入 fixture 中选定的共同 base 参数；
- 明确共享的只读参数组，用于验证多使用点的梯度累加。

每个键在 fixture 中声明期望为 connected、disconnected 或 structurally absent。connected 键即使导数数值为零，也必须产生并比较同 shape 零 Tensor；`None` 只接受于 disconnected 或 absent 键。梯度先检查 finite，再使用第 4.2 节相应 backend/dtype 的 comparator。

至少包含以下隔离目标和路径断言：

1. post-update + BO 中，仅由 selector logits/probability 构成的目标对 proposal 和 Update 参数具有按解析公式预期的梯度；默认不得 detach proposal；
2. pre-update 的同类隔离目标不经本 Token proposal 返回 Update，但 active NodeCompute 仍可经已提交状态返回 Update；
3. `EMIT-HST` 前向逐元素等于 \(g\)，且对 active probability 的主任务局部导数为 \(\zeta^{\mathrm{ST}}(g-h)\)；`EMIT-HARD` 没有这条 probability 梯度；
4. candidates、availability 基准、Top-K IDs、active set 和默认 selector-history 写回 stop-gradient；
5. inactive receiver 的 NodeCompute 参数不从主任务目标获得梯度，selector 参数仍可从辅助项或 Hard-ST 的已选路径获得声明的梯度；
6. chunk detach 阻断前一 chunk 状态到后一 chunk 目标的反向边，采用相同 forward state carry 的 no-detach 对照则保留它；
7. `BAL-AVAIL-SOFT` 只通过 \(P_v\) 中的 probability 返回梯度，\(N,A_v,F_v,Q\) 均 stop-gradient。

连续梯度检查使用 margin-safe route 或固定 route。CPU FP64 的解析导数、`gradcheck` 或方向有限差分至少提供一种独立于 executor 差分的证据；离散成员关系本身不做连续导数声明。

optimizer fixture 保存 optimizer 类型、全部超参数、参数组顺序和初始 optimizer state。比较一次 step 前的梯度、step 后参数和 optimizer state；只比较 loss 而不比较更新结果不足以通过 optimizer tier。

## 6. Chunk、mask 与状态事务

对长度 \(T\le6\) 的 deterministic fixture，测试枚举全部 \(T-1\) 个边界的切分子集；更长 fixture 至少覆盖单 chunk、逐 Token、奇偶交错长度和包含空尾部的四种切法。每种切法从同一初始状态开始，并比较逐 Token 输出、exact trace 中的语义事件、最终规范状态和下一位置。

LM loss 跨 chunk 只合并负对数似然总和与 LM target 数。`BAL-AVAIL-SOFT` 按实现计划第 6.3 节合并每个 site-region 的 \(N,P_v,A_v,F_v,Q\)，完整窗口结束后才做除法、平方和 region reduction。测试同时比较这些原始统计、最终 loss 和 VJP；各 chunk mean 的平均不作为合法合并方式。

mask fixture 至少分别包含：

- prompt Token：execution 为 true、LM target 为 false、routing stats 使用默认 true；
- padding：三个 masks 都为 false，输出旁路且状态不变；
- 自定义 routing-stat 子集：前向 route 与状态不变，仅辅助统计变化；
- 不同 batch rows 的不等长序列和调用间 row 重排；
- 非法的 LM/routing mask 超出 execution mask。

状态生命周期 fixture 覆盖创建、连续 chunks、site-local reset、顶层 all-site reset、显式释放、重复 `sequence_id`、位置重复/倒序/跳号以及并发写冲突。所有非法情况必须在公开状态发布前失败。

标准事务 fixture 让较早 Token 成功写入 staged state、较晚 Token 的 `requested_k` 越界或局部操作显式失败；调用后逐键比较外部状态、selector-history 和下一位置与调用前完全相同。另用测试专用故障注入破坏“active receiver 发送”不变量，验证空终端防线，但不能把这种注入样例计为合法标准 Plan。失败调用不得留下部分 output、部分充分统计或可被 checkpoint 捕获的半提交状态。

## 7. 覆盖门槛

### 7.1 确定性手工语料

资格测试至少长期保存以下 fully specified fixtures，并为每类至少提供一个 exact trace：

| 类别 | 必须覆盖的语义 |
| --- | --- |
| singleton forced-active | 最小输入/终端、实际 Read/Score logit、\(p=1\)、active 不依赖 Top-K、identity |
| 单层 \(R=2,8\) | Top-1、Top-2、all、输出均值 |
| chain | 跨 region 顺序、状态 carry |
| diamond | fan-out、关闭父边、fan-in 与 edge order |
| unequal-path | 短路径缓存和 skip edge |
| multi-entry/multi-terminal | 边界广播与终端聚合 |
| mixed regions | 同层独立 regions、singleton 与竞争并存 |
| forced backbone | 可选分支全关但仍有终端输出 |
| injected empty-terminal invariant | 测试专用故障注入、调用失败和事务回滚；不属于合法标准 Plan |
| small expanded HB | Lines、barrier、tree/local/shortcut/mirror 标签 |

这些 fixtures 合计覆盖 N/content、SD/content、SD/pre、BO/content、BO/pre、BO/post；hard、Hard-ST、soft probability Emit；mean、learned convex、edge-affine Aggregate；none、历史、EMA、Gated DeltaNet 和规范窗口 Attention 状态；固定与运行期 K，其中至少一个下游候选为空的事件携带超出值域的整数占位值，并证明该位置没有被读取；独立与共享只读参数；零、固定非零和可学习首状态。

对所有受约束的配置轴做 pairwise covering：每一对合法取值至少共同出现一次。无效组合不为了覆盖而执行，而是作为 validator failure fixture 保存明确错误类别。

### 7.2 随机与非法语料

一次 CPU 资格运行的最低门槛为：

- 256 个固定 seed 的合法小 Plan fixtures，全部执行 CPU FP64 与 FP32 的逐 Token/通用 packed 差分；
- 其中适用结构的全部 fixtures 执行对应特化路径，不能只挑已知通过样例；
- 至少 64 个 backward/VJP fixtures 和 16 个 optimizer-step fixtures；
- 至少 96 个非法 Plan 或运行期输入 mutants，覆盖 cycle、region 内边、重复边/ID、shape/dtype role、formula config 未知键/缺键/改参数 schema 的值、状态别名（同一 Tensor object 与同 backing storage 的不同/不重叠 views）、K、mask、位置和事务错误。

随机合法集合还必须达到以下计数，而不只是总数：每种手工图 motif 的随机变体至少 16 次；六种标准 profile/selector 时序组合至少各 16 次；每种 Aggregate、Emit、state 和 K 类至少各 16 次；exact tie 至少 16 个事件、margin-safe 至少 64 个事件、near-boundary 至少 16 个诊断事件；较晚事件因非法运行期控制或局部操作失败而整调用回滚至少 16 次。空终端不变量的故障注入另计，不能贡献合法 Plan 数量。一次 fixture 可以贡献多个不同计数，但同一个事件不能通过重复读取被重复计数。

NPU FP32 资格集合从同一 CPU bundles 中选取至少 64 个 forward/state/route cases、32 个 backward/VJP cases 和 8 个 optimizer/checkpoint cases，并保证每个声称支持的核心算子、空段/尾块、非连续布局和边界 shape 至少出现一次。数量门槛不能替代实际 operator coverage。

随机测试只在固定 seed 列表、总数和各覆盖计数同时达到且零失败时停止。失败 bundle 必须保存原始 fixture，并尽可能收缩为仍失败的最小 Plan；收缩结果不能取代原始复现 artifact。

### 7.3 专项范围

HB executor 测试消费 fully expanded Plan。Builder 若要形成能力声明，另行固定 Builder 名称、版本、config、展开端点列表和 golden logical Plan hash；只验证一个展开 Plan 不能证明 Builder 的所有配置。

Flat MoE 和 Dense 不属于 SettleGraph executor 资格集合。科学实验若使用这些基线，另建 suite 覆盖 expert/gate 合并、capacity、drop/reroute、辅助 loss、梯度与 checkpoint。Base Qwen 接入 suite 另行覆盖四种 placement、causal attention mask、position IDs、KV cache 的 prefill/decode、最终 logits、LM loss 和共同 base 参数梯度。

## 8. Checkpoint 契约

checkpoint schema 至少记录：

- schema 版本、logical Plan 与 logical Plan hash、保存时 concrete execution binding 与 typed Plan hash；
- base 与 SettleGraph 参数、可学习首状态及参数共享关系；
- optimizer、scheduler、AMP scaler；
- 按 owner 规范化的 receiver state、selector-history、Attention 有效窗口和每个序列的下一位置；
- global step、Token 计数、epoch、gradient-accumulation microstep，以及允许中途保存时的累积梯度；
- sampler/data cursor、data identity、worker/采样状态；
- CPU 与所选 backend 的适用 RNG 状态、随机键版本、确定性和精度设置；
- 尚未归约的 LM/balance 充分统计量与窗口位置；若承诺窗口中途 exact resume，还包括可重建其 VJP 的 replay 输入或梯度算法状态，否则 schema 必须只允许在统计窗口边界保存；
- base/tokenizer/Builder 身份、仓库与配置身份、checkpoint 内容 hash。

`init-from` 只加载记录中声明的权重和可学习首状态，重新初始化 optimizer、进度、RNG、receiver state、selector-history 与序列位置。`resume` 加载 schema 声明的全部状态；若任一必需键缺失、shape 不兼容或 Plan 身份不符，必须在修改输出目录或外部状态前失败。两者互斥。

checkpoint 测试至少包括：

1. 同栈新进程保存—加载，加载前后 forward/state/VJP exact 或在相应 dtype tolerance 内一致；
2. 中断点前后拼接执行与无中断参考比较，包括数据顺序、optimizer 和充分统计窗口；
3. `init-from` 证明 optimizer、RNG、进度和序列状态没有被静默恢复；
4. CPU 保存的 portable checkpoint 在另一个已验证 backend 上 handoff，并按 parity 门槛完成下一步；
5. corrupt hash、错误 Plan、缺键、额外键和 dtype role 不兼容均明确失败；
6. checkpoint 不能观察到失败调用的 staged state。

跨 backend handoff 只承诺可装载和在声明容差内开始新的正确轨迹，不承诺 RNG 或长期训练轨迹相同。exact resume 只对记录的同 backend、软件栈、数据顺序和确定性条件成立。

checkpoint 必须在没有进行中调用事务的原子安全点发布。若训练配置让 autograd 图跨保存点延续，exact resume 还需用保存的可重放前缀重建该图；否则合法保存点必须限制在声明的 detach/backward 边界，不能只序列化状态 Tensor 后声称恢复了同一反向轨迹。

## 9. Capability 与证据契约

一个 capability cell 至少由以下坐标唯一确定：executor、局部算子 formula/实现变体、backend、host architecture、accelerator case 的精确 SKU、dtype、eager/compiled/custom 模式、forward/backward，以及必要的 shape/layout 范围。`reference`、`packed`、`optimized` 描述 executor 能力级别，不替代支持状态。

支持状态只使用：

- `planned`：目标在范围内，尚无实现声明；
- `implemented`：存在代码，但没有该精确目标的完整通过证据；
- `verified`：该 cell 要求的测试层级在记录的精确目标和 commit 上通过；
- `unsupported`：入口主动拒绝，并记录稳定原因。

一个 `verified` cell 的机器可读证据至少包含：契约版本、日期、commit/dirty 状态、完整命令、fixture 集 hash、logical/typed Plan hash 集、resolved backend 与 `resolution_reason`、host architecture、设备精确 SKU/逻辑索引、Torch/TorchNPU/CANN/driver 等可观测版本、dtype/精度/确定性设置、executor 与 kernel/compile 选择、comparator 结果、artifact hash 和通过的层级。

CPU cell 至少通过 static/CLI、CPU semantic、forward、适用的 backward/optimizer 和 checkpoint round trip。NPU cell 还必须通过：

1. 独立新进程中的真实 allocation、项目所需算子、同步和 CPU copy；
2. 与序列化 CPU fixture 的 FP32 parity；
3. 项目算子的 forward/backward、边界 shape/layout；
4. optimizer 与 portable checkpoint handoff（训练范围内）；
5. profiler、dispatch trace 或当前栈支持的等价证据，确认关键操作实际在 NPU 且没有未经声明的 CPU fallback。

只出现 NPU output Tensor、包版本或一次 allocation 不能证明关键计算位于 NPU。若当前栈无法可靠观察 fallback，相关 feature/workload cell 保持 `implemented` 或 `planned`，不能标为 `verified`。普通测试运行可以跳过未请求的可选硬件；显式请求 `npu`、某 dtype、executor 或 feature 时，不可用必须非零失败而不是 skip 或回退。

CPU、CUDA 和 NPU parity case 分别在新进程解析 backend 后运行。进程间只交换 CPU fixture/result artifact；不能在已经建立 CPU autograd 图的进程里晚加载 TorchNPU，也不能同时导入多个 vendor family 后把结果当作干净证据。

## 10. 最小通过报告

一次资格报告至少回答：

- 哪些 logical/typed Plans、executors、operators、backends 和方向通过；
- fixture、随机 seed、覆盖计数和失败收缩 artifact 在哪里；
- exact trace、VJP、optimizer、chunk、事务与 checkpoint 各自的结果；
- 使用了哪些容差，最坏误差位于哪个稳定路径；
- NPU 关键操作的 placement/fallback 证据是什么；
- 哪些 capability cells 仍为 `planned`、`implemented` 或 `unsupported`；
- HB Builder、Base Qwen、Dense/MoE、低精度、distributed 或 optimized path 中哪些不在本次范围。

只有报告引用的 artifact 实际存在、hash 可验证且对应精确 commit/环境时，才能把 cell 标为 `verified`。文档审查、静态扫描或尚未在目标硬件运行的代码都不能单独产生该状态。
