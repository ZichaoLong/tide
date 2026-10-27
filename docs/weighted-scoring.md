# 带权局部评分、区域匹配、证据与 key

固定 [区域教材](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)
`0770593` §3.1–3.6、附录 S.1。输入/状态的含义、Read 三快照、Full 与共享绑定沿用
[共同契约](weighted-message.md)；[发送函数](weighted-sending.md)只消费本页实际描述。
记忆来自本次选定的 KV/affine/delta/GDN/SSD 的 Mem；不替换或重放状态。

## 配置、域与返回

`RegionalConfig` 提供 local_score=cat/match、region_score=ind/att，
context 的 region_evidence=cat/grouped，att 的 region_key=content_stats/content。
这四个选择分别绑定函数，互不暗含；fixed 控制不使用 evidence 函数。
`selection_width` 是 dsel，match/att/grouped 任一启用时默认 4；均不用时不接受该字段。
hidden_width=dr，region_control_width=dreg 的旧含义保持。

match 使用非负 alpha_match/alpha_res，默认各 1；att 使用非负 alpha_loc/alpha_sel，
默认各 1。对应函数关闭时不接受其尺度。att 的 key 默认 content_stats；
节点增益描述的 query_weight=mu/received，默认 mu；regional 描述固定 received。
ind 不读取 key/query/区域匹配尺度。Python 配置解析默认后冻结，C++ 配置显式验证所赋字段。
所有选择、维度、尺度进入图身份；没有这些新字段的既有图身份保持。

Mem 返回 `Mshort,Mlong:[ds]` 与 `Wshort,Wlong:[]`；当前聚合为 `r:[],u:[d]`。
具名描述严格采用适用布局：

| 绑定 | 字段与尺寸 |
| --- | --- |
| regional / ind | w、mu、score 标量；evidence:[dreg] |
| 节点增益 / ind | w、mu、score 标量 |
| regional / att | 上述四项加 x:[dsel]、key:[dsel] |
| 节点增益 / att | 上述三项加 x:[dsel]、key:[dsel] |

score 字段始终是局部评分，最终区域评分由区域函数实际计算。
字段根保持独立，checkpoint/trace 仅解码实际计算结果，不借用参考递归补字段。
固定控制的零 evidence 是该描述定义的一部分，节点增益描述没有 evidence。

## 函数与参数对应

Python [scoring.py](../python/tidegraph/weighted/scoring.py) 和独立 C++
[weighted_scoring.cpp](../cpp/src/weighted_scoring.cpp) 实现下列数学映射。
Python 参数名的 read_program 前缀对应 C++ NodeWeights::extra 的同名路径。
参数默认有限高斯初始化标准差 .15，节点/部件独立；eps 来自该节点记忆配置。

| 数学条目 | 符号及参数维度 | 输入 → 返回 |
| --- | --- | --- |
| §3.1 式 (1) | ScalarHead；matrix:[dr,d+4ds+3]、hidden_bias/vector:[dr]、bias:[] | 共同 cat 特征 f → 局部实数 |
| §3.2 式 (2) | Encoder；in_weight=B:[dr,dg]、hidden_bias=b:[dr]、out_weight=C:[dsel,dr]、out_bias=a:[dsel] | z → N(C SiLU(Bz+b)+a) |
| §3.2 式 (3) | MatchScore；独立 encoders.cur/mem/stat；gate/up:[dr,3dsel]，各偏置:[dr]，vector:[dr]，bias:[] | 三组编码，cur·mem/sqrt(dsel) 加门控残差 → score |
| §3.4.1 | GroupedEvidence；独立三组 encoders；cur/mem/stat/interaction:[dreg,dsel]、bias:[dreg] | 三编码与 cur*mem 经四投影之和、归一化 → evidence |
| §3.6.1 | Describe.current_weight:[dsel,d]、current_bias:[dsel] | N(u) → x，未额外归一化 |
| §3.6.2 式 (4) | content_stats：key_weight:[dsel,2ds+2]、key_bias:[dsel] | 两记忆与 log1p(Wshort/Wlong) → key，未归一化 |
| §3.6.2 式 (4) | content：Describe.key 的 Encoder；dg=2ds | 两记忆向量 → 归一化 key |
| §3.6.4 式 (5) | Query；in_weight/out_weight:[dsel,dsel]、两偏置:[dsel] | Σweight*x/max(1,Σweight) → Wout SiLU(Win x+bin)+bout |
| §3.6.4 式 (5) | RegionalTotal.scores / C++ region_scores | alpha_loc*局部分 + alpha_sel*query·key/sqrt(dsel) → 每候选最终分 |

三组输入为 cur=N(u)、mem=[Mshort,Mlong]、stat=[log1p(r),log1p(Wshort),log1p(Wlong)]。
Describe.inputs 从一个实际 Mem 快照取得它们，cat/current/score/evidence/key 使用这份输入。
content key 不强制零记忆映射为零，因为编码含偏置；其范数不超过 sqrt(dsel)。
grouped evidence 范数不超过 sqrt(dreg)。query/key 的匹配不是另一次 Attention 份额归一化。

## 依赖、共享和反向

Read0/-/+ 分别绑定初态、旧态、proposal；这里每个函数使用该次 Mem 的实际结果。
KV qry、循环 state read 本来就可能通过查询依赖 u；Read+ 还通过写入依赖当前接收。
所以 content key 对统计量“无直接依赖”只指固定两记忆向量后的函数，不删除这些间接路径。
局部分、发送上限、query 权重与区域证据仍保留各自的统计依赖。

regional query 始终使用 received，改变节点增益不暗改这一规则；节点增益 att 可以显式选 mu。
零权重候选对 query 的值贡献零，所有真实候选的 key/局部分仍按所选函数存在。
alpha_loc=0 或 alpha_sel=0 使用普通乘法的连接零梯度；描述仍保留该局部分/key。
alpha_loc=1、alpha_sel=0 只证明最终评分的数值投影等于 ind，不能声称完整记录等价。

一维 Encoder/Query、MatchScore 的残差 gate/up 与 GroupedEvidence 的四个投影
采用矩阵乘列向量的教材顺序，与 C++ 的浮点归约一致；
任意 leading axes 保留批量求值。逐标量公式与方向差分继续独立核验数学映射。
所有映射使用标准 ATen/PyTorch 一阶链式反向；发送映射自己的精确/排序分支 VJP 另见发送契约。
共同阈值通过分配的显式 score_shift 参数回传，以解析的活动总量约束零方向避免伪梯度。
score、evidence、key 编码默认互不共享参数。允许显式同形 Parameter/Module 绑定；
相同参数和相同输入下各使用者的反向贡献必须汇总。当前重复调用没有声称消除了共享计算。
同一区域的其他参数共享与各节点私有记忆状态保持独立；checkpoint 校验/恢复别名关系。

## 独立验证和运行边界

[公式测试](../tests/test_weighted_scoring.py)提供：

- 归一化编码偏置、match 内积/门控项、grouped 四投影的手算及独立偏置导数。
- 不复用 Encoder 的逐标量矩阵求和/SiLU/RMS 对照，逐输出和全部参数/输入根。
- 两种 key 固定记忆向量时的统计依赖、非零初始 key；query 的 received/mu 手算与独立字段根。
- 关闭区域匹配的数值投影对照，连接零梯度和不适用字段拒绝。

[整图测试](../tests/test_weighted_scoring_execution.py)覆盖代表性配置组合与五种异构记忆，
完整字段/输出/状态/控制/pending/独立根；FC/HB、PDG 反馈、三种 Read 和每切点续接。
三步 AdamW、共享 score/evidence/key 编码、checkpoint/optimizer 恢复后再更新、
全参数和输入方向差分验证训练链。不同函数的属性检查与同函数完整记录比较分别列明。
[standalone C++](../cpp/test/weighted_scoring.cpp)另有局部手算及 VJP。
实际开发和不可变资格结果见 STATUS；不把编写了测试当成已通过。

本增量不自动认证 joint/packed/VJP 优化、计算缓存、设备常驻、跨卡或性能；
standalone 局部公式不替代其完整训练矩阵。公共调用与后端资格仍分别记录。
