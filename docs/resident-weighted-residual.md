# 常驻 Full 残差与 SwiGLU

固定 [KV §4.2](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
及四循环 Full 的共同内容链，由 `weighted_residual.h` 实现：
`x=u+alpha_memory*Wout*m`，
`G=x+alpha_ffn*Wdown*(SiLU(Wgate*N_d(x))*Wup*N_d(x))`。
本层不生成发送强度，也不把 G 乘 c；(c,G) 的完整消息由图层展开。

私有 `input[R,d]`、记忆读出 `memory[R,dm]` 与动态 `owners[R]` 分开。
四个参数 bank 的维度为 `out[B,d,dm]`、`gate/up[B,df,d]`、`down[B,d,df]`，
没有额外偏置。owner=-1 表示没有正式 Full，不能读取该行输入或参数。
同一参数 bank 的所有使用者贡献汇总，跨角色共享仍交由 canonical 参数归约。

`append_weighted_residual` 使用 vendor RMS/BMM/SiLU/multiply/add；两次相同的 RMS
属于本层计算选择，不改变函数或参数共享身份。有限检查和实际行 mask 在设备完成。
`append_weighted_residual_vjp` 从独立行根回传 input、memory 与全部四个参数角色；
alpha=0 仍保留数学图的连接零梯度。保存前向实际版本，反向不重读原参数/输入。

显式 FP32 accelerator，scratch 在提交前准入；输入/中间值非有限明确拒绝。
本层的 memory 由上游实际记忆函数提供；常值空记忆的 None/零由该函数的 VJP 处理。
它不替代 KV Attention、循环读取、Describe、图推进或 checkpoint。

`weighted_residual_check.cpp` 用独立逐行公式和 autograd 比较完整前向、所有私有根、
四个共享参数角色、None/零；四组零/非零残差尺度、三个 bank、两次重放48行，
含未使用的 poison bank/行、零 cotangent、保存版本及容量拒绝。
公共入口 `scripts/{build,verify}_device_control.py --checks weighted-residual`；
CPU 公式基线、NPU 实机和 CUDA 构建/真机各自记账。

固定源码资格见[7d86d9e运行证据](evidence/core5-resident-kv-full-20261027.md)。
