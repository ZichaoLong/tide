# Weighted FP16原始状态、投影与固定时间卷积资格

固定实现`2e4d445e49118d10b0d02a13f19379c99ff15478`，固定上游`0770593`。
[机器记录](core5-resident-half-primitives-20261028.json)保存源码、binary/loader、实际检查及原失败摘要。
契约分别见[原始状态](../resident-weighted-state.md)、[共享投影](../resident-weighted-projection.md)、
[固定时间卷积](../resident-weighted-convolution.md)。这是三个局部组件资格，不是完整FP16图/训练交付。

NPU实际通过六项组合：三个组件各自FP32与FP16。FP16是实际输入/参数/结果存储，
厂商算术、保存值和全部伴随明确为FP32；整数元数据保持int64。投影与卷积仅在最终输出存储处舍入，
没有宣称使用FP16 GEMM或复现CPU半精度叶的逐步梯度舍入。Cast使用声明的一阶identity VJP。

- 原始状态：每dtype五schema、40行重放；原始时间缓存、弱/零/饱和接收、每字段独立VJP、
  改写来源后的反向、容量/非法拒绝及失败不提交。FP16完整逻辑字段按位比较；提交另外保留
  signed zero、subnormal、padding NaN payload及大于2^54/MAX的int64位模式。CPU codec三dtype通过。
- 投影：每dtype有/无RMS×有/无bias四配置、48行重放、三个共享bank，完整输入/参数根和保存版本。
  独立逐行FP32公式在实际量化输入上求导；FP16前向只在最终输出舍入。
- 卷积：每dtype两种tap数量、20行重放、三个共享bank；独立dense原始时间信号+grouped conv1d
  核验完整缓存/当前输入/参数VJP。raw update→新快照卷积→两层VJP保留质量的connected zero。

投影与卷积的最小半精度次正规数及舍入下溢零另用精确断言检查，FP32伴随和连接不因数值零消失。
有限FP32输出溢出FP16存储会拒绝，卷积核对live raw owner未提交；FP16 cotangent明确拒绝。
CPU相关32项通过、无skip。实际NPU为物理9→npu:0，LibTorch-NPU2.10/CANN9.0。
源码干净、持久passed/exit0与systemd inactive/success/0相互核对。

CUDA同三个组件的build/loader/help及CPU codec通过。Ninja实际所选依赖包含12条CUDA编译命令，
所有160条已配置CUDA编译命令均请求80/90/100；不把未选择的目标说成已构建。
没有GPU真机、Python-owned FP16运行或性能结论。

原失败未改写：初始CPU codec正确拒绝放在零适配字段中的测试次正规数，后移至keys/H；
两次CUDA launcher错误发生于编译前。首次NPU数值资格的FP32通过，FP16因check未显式启用
RuntimeOptions准入而在运行前拒绝；相同问题的另一构建主动取消，保留cancelled/143。
修正入口后本报告才完成两dtype实际运行，未改数学输入、容差或严格支持条件。
复现使用机器记录中的官方build/verify命令，全部源码、测试和契约位于固定提交。
