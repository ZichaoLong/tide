# KV 数值块：一次打包、整组计算、独立发布

固定数学见[KV §3–4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)和
[描述/区域分配](weighted-scoring.md)；采用版本由[semantics](semantics.md)固定。
本页定义数值接口，不要求图执行器采用某个历史名称、放置分层或事件对象。
公共窗口仍见[张量窗口](weighted-tensor-windows.md)。

## 数值和所有权

相容节点的查询输入为 `[G,B,Q,d]`，历史为 `[G,B,K,d]`，各投影参数为
`[G,out,in]`。真实相同参数可保留一个 `[out,in]` 矩阵；等值独立参数仍有不同根。
输入历史只打包一次，不把矩阵或历史复制成每个query的私有bank。
正写入准备在 `[G,1,R,d]` 上计算base、低秩因子和summary；R可以不同，
零写入节点保留原状态，不参与准备数值块。所有归一化、Read查询/两种时间权重、
Describe头/证据/key和Full残差在这些张量上计算。
短、长Read共享一次QK映射；其加权归一化与伴随贡献分别计算。

不同Q/K长度允许进入同一组。真实行域、padding和位置来自int64布局；零质量的真实接收
不是padding。加入节点后按全组新的最大Q/K重新计算空间预算，不能简单累加各节点未填充预算。
状态仍由独立的实际写入bank和末态保存；查询的有效投影不能替代raw checkpoint。
Read0取显式绑定初态，Read-取本时刻之前，Read+和Full取proposal可见快照。
Full只处理正control的实际行，返回未乘c的G，发送强度另行交接。

共同原点位置仍遵循 `bounded-common-origin-v1`：先以int64减去原点再转换，
频率/整数跨度在声明域内才分离Q/K旋转；域外保持相对位置算法，不静默改函数。
FP16的既有实现和位置资格保持独立；不由FP32/FP64新数值块结果推断half通过。

## 反向边界

`tensor_kernel`在无梯度装配区建立每个角色的一份packed leaf，并执行一个私有数值图。
图的大小随有限公式增长，N/B/Q/K扩张是张量尺寸增长。边界原始参数、状态、输入各自保留
版本守卫和独立身份；一次块反向归并各节点、字段及全部alias使用者。

每字段到角色的结构依赖由实际私图追溯；节点根为None时不制造零连接，显式零bar仍为连接。
该规则要求组内节点计算独立且结构分支一致，因此qry实际有/无可见正权重先分组。
内部多输出custom Function不能仅以所有输出的输入边并集推断精确字段依赖。
sum的空张量归约、Full空记忆的out连接零、零alpha分支均保留原来的连接契约。
共享角色伴随只回传一次，由原始alias身份接收累计贡献，不乘以角色重复次数。

普通列根与逐事件根是不同观察接口。比如空旧态列与后续写入拼接后，可有connected-empty根；
独立标量事件没有使用那个空槽时可能为None。数值/空域对照、精确列根对照和完整事件资格分别检查。
需要逐事件因果根的客户端仍应显式请求该观察能力；它不强制普通训练内部物化所有事件。

## 实现、检验与诊断

Python入口 `kv_columns_state`/`kv_column_groups`按能力调用 `kv_tensor_prepare`/`kv_tensor_groups`；`kv_tensor_layout`负责布局和角色，
`kv_tensor_formula/tensor_describe`负责数学，`tensor_kernel`负责一次打包/发布及一阶反向。
独立C++入口的同一职责由`weighted_kv_tensor`、`_layout/_positions`、`_math`、`_vjp`承担，
不依赖Python调度或Python私有图。两种客户端分别构建和验收，不能相互替代资格。
不调用独立基线或逐节点数值helper补写输出。原单节点/half/域外位置路径是明确能力分支，
执行失败、容量错误或非有限值不触发静默后备。

- `test_tensor_kernel`：独立标量及映射矩阵验证None/零、alias、冻结、版本、ragged转置、重复反向。
- `test_kv_tensor_batches`：不同Q/K同组、真实padding预算、独立标量快照及完整列根。
- `test_column_query_dispatch`：有限Describe候选逐字段根、一般默认入口和空记忆连接。
- `test_settle_columns/test_column_positions`：独立调度、端口/消息/raw、原时间及精确续接。

私图注册到现有 `autograd_census`，诊断入口 `profile_lh_tensor.py --census`，与旧外层图统计分列。
`packing/bucket/kv_tensor_read`、`kv_tensor_full`记录实际数值组；投影区分整组BMM和真实共享GEMM。
调用次数、私图节点数、字节和完整耗时是不同证据。构建、真机、规模性能及新进程恢复需要
固定源码资格；当前未完成事项只保存在[ROADMAP](ROADMAP.md)。
