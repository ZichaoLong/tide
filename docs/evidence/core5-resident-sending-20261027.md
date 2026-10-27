# Core-5 常驻发送原语资格 — 2026-10-27

固定实现 `8e7298d181bca157def003fc898cea60b869fbee`，固定上游 `0770593`。
[命令、环境、摘要](core5-resident-sending-20261027.json)；[函数/反向契约](../resident-weighted-sending.md)。
NPU与CUDA构建资格均clean、exit0、已终止。

| 路径 | 已通过范围 |
| --- | --- |
| CPU语义源码 | 34项，2.55秒；独立exp商与方向差分、LP顶点、分数发送反例、零base右导数/全零map、比例原点、大int64 ID、独立根/None与零、padding/严格拒绝 |
| standalone NPU FP32 | Direct两种base、Unit两种base、Pooled、Proportional、SortedFill、Hard共8配置；96物理行/80非空帧，完整五输入VJP、同program两组输入/长度重放、失败不提交/sticky error/容量拒绝 |
| CUDA构建/loader | sm80/90/100的前向、VJP、错误汇总kernel与standalone检查程序通过；无GPU实跑 |

NPU为LibTorch2.10/CANN9.0、动态单张Ascend910_9392；参考是独立C++标量公式/autograd，
atol1e-6/rtol1e-5、ID/支持/None与零严格。CPU适配不冒充实际设备。
参数k约束强度与硬选择数量、候选存在性与零base、不同map/分支导数均保留独立身份，未使用HST。
前向/反向在设备程序内，单个scalar worker拥有紧凑行；不声称吞吐优势或完整图packed执行。

NPU复用build2，但资格前验证完整component/core/binary摘要与固定源码一致，保留原manifest身份。
首次开发因AscendC不允许隐式bool→float编译失败，CPU63已通过；改显式1.f/0.f保留同一掩码。
原failed/source快照保留。CUDA为新构建，Torch2.10+cu128/Toolkit12.8.1，loader闭包通过。

这些原语与已验收Alloc仍需接入区域head/query、完整图在线推进、retained训练、optimizer、
精确checkpoint和跨卡；该报告不认证这些范围、FP16载荷或性能。GPU真机单独待验。
复现按JSON和[development](../development.md)，新输出目录，不依赖旧运行目录。
