# Core-5 带权 eager FP16 NPU 资格 — 2026-10-27

固定实现 `01689c00da713413af494b8be0b42d2fe12b59c3`，固定上游 `0770593`。
所有下列固定任务clean/exit0、服务inactive/success；[命令/配置/环境/输入及日志哈希](core5-half-device-20261027.json)。
实现与[half数值CPU资格](core5-half-20261027.md)衔接，范围不是完整FP16开关矩阵。

| 入口/路径 | 独立检查通过范围 |
| --- | --- |
| CPU入口 | 12项：Python/native完整记录/独立根/四步/恢复与早期精度拒绝 |
| Python及native，各自mixed-c | KV-FC、affine非对齐DAG、delta反馈PDG、SSD-DAG：显式AdamW，分别4组合 |
| Python及native，各自mixed-c | GDN-HB、异构Attention/Bernoulli：显式SGD，分别2组合 |
| Python及native，各自mixed-a / mixed-b | KV-FC / delta反馈PDG：显式AdamW，分别各1组合 |
| Python及native，各自两卡mixed-c | KV-FC，显式ASCEND_LAUNCH_BLOCKING=1及AdamW，分别1组合 |

payload/参数/raw state为FP16，评分与masters为FP32；loss_scale=16，lr=.001。
两个独立CPU/候选拥有同一量化参数及输入，CPU标量轨迹不传入候选。比较完整返回、
实际owner、streaming/online prefill、descriptor/control/output及原始浮点State独立根、
None/连接零、保留图切点、三次更新；从实际raw/checkpoint解码后做第四次更新。
同owner、CPU handoff及双卡交换owner均检查。恢复raw/master/slots逐值精确；
数值计算atol1e-3/rtol2e-2适用于所有浮点观察量，整数/存在性/路由严格。
输入文件hash逐案例保留；没有为设备改输入、候选集合、平票规则或阈值。

两卡是单进程区域分置与真实跨卡消息，不是HCCL数据并行或resident推进。
显式同步成功不解除既有默认异步CANN invalid event；half默认异步未据此认证。
GDN默认AdamW的CPU严格见证仍在，SGD单列通过不等价于其修复。
本报告也不解除SDK2.10 fc/affine的FP32严格见证，不认证standalone half或CUDA。

native half mixed-c异构SGD profile，单独进程一次warmup和一次完整训练观测：
35945 kernel（29803 AIVEC、4234 MIX_AIV、1888 AICORE、20设备AiCPU），
1222 host scalar、1368 memcpy；32 candidate、31 Full、44 messages。
计入输入准备/上传、调度、packing、forward/loss、backward/finite、master SGD及同步。
未观测到host fallback事件/日志；AiCPU排序与host fallback分账。有限trace可见性
不证明任何形状所有算子都无回退；未控制外部争用，不把kernel数当吞吐或性能排名。
与既有FP32 AdamW profile的optimizer和算术不同，不能据此推断精度带来的加速。

Python3.12.13/Torch2.12.0+cpu/Torch-NPU2.12.0.post2/CANN9.1，aarch64 Ascend910_9392。
root持有共享目录、padding/base-format、AiCPU int64排序及TypedStorage警告保留；
本批日志未发现fallback/error。两设备由队列动态分配；请求/生效及映射均在JSON。
NPU build来自开发tree cfee98d5，其C++摘要及全部binary摘要与01689c0一致；原source字段保留。

复现：检出固定实现，按[development](../development.md)独立构建匹配NPU Python库，
使用JSON中的项目check/profile命令和新输出目录；无需旧运行目录。两卡显式设置记录的同步环境。
各任务background.slice/Nice10、ATen/BLAS单线程，CPU600秒/NPU1200秒上限。
所有raw结果在artifacts/jobs/core5-half-*-qualify；正式报告不依赖它们继续在线。
