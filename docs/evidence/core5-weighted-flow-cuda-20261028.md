# Weighted 常驻图 CUDA 构建与安装资格

固定实现 `8eebceccc8e209a6f5f08caf4d262a4e388d280b`，上游 `0770593`。
[公共契约](../resident-weighted-flow.md)；[机器记录](core5-weighted-flow-cuda-20261028.json)。

Torch2.10/cu128、CUDA12.8.1、aarch64：完整图、配置、设备services及安装消费者构建/loader通过。
检查159个已配置CUDA翻译单元的编译命令，实际架构集合均精确为80/90/100；
这不是声称159个单元全部属于所选构建目标。8项架构策略检查（0.12秒）、CPU跨DSO检查通过。

构建规则在加载Torch前固定其legacy FindCUDA架构，禁止继承wheel的sm50等额外默认值。
显式代码要求80及以上，拒绝native/all及不一致的real/virtual后缀。
旧开发sm50编译失败保留；新开发与固定资格均通过，不覆盖旧失败。

资格任务clean、inactive/success/exit0；保留原manifest，Ninja按源码与命令依赖重编，
所有选定二进制/loader与安装消费者摘要记录。600秒上限、2构建线程、BLAS1。
无NVIDIA GPU，本报告仅认证源码构建与加载，实际GPU前向/训练/续接/跨卡/性能仍待验。
