# Core-5 CUDA 构建与 CPU 资格 — 2026-10-27

固定实现 `670ead324744bfb45b006d9ef3a22d1572ed0829`，固定上游 `0770593`。
[源码、环境、命令与摘要](core5-cuda-eager-20261027.json)。干净源码任务已终止，exit0。

私有 aarch64 Torch2.10.0+cu128 / Python3.11.15 / Toolkit12.8.1 构建成功。
C++ core、standalone 检查入口和 Python native adapter 均链接完成。
四个 CPU 门禁入口共 **24 项通过**，143.99秒：master optimizer、普通optimizer、
精确checkpoint和完整带权图训练。FP64/FP32及显式half master案例的范围与
[独立SDK资格](core5-native-master-20261027.md)中的CPU入口相同。
本次验证这个CUDA链接工具链在CPU上的实现与恢复，没有运行CUDA张量。

实际 `torch.cuda.is_available()==false`；`tidegraph-runtime-check --device cuda:0
--dtype float32` 返回2，明确报告设备不可用，没有静默改到CPU。
本机没有NVIDIA GPU，GPU算子、mixed/resident、训练/恢复、跨卡和性能保持待验。
这次core构建也不替代专用CUDA resident kernel编译。

首次服务因PATH中没有Ninja而配置失败；第二次因launcher准备中断而exit127，
未运行workload。两项failed记录保留；本次使用项目私有Ninja并保持CUDA Python优先级。
公共工具链未修改。background.slice/Nice10、2构建线程、ATen/BLAS单线程，1800秒上限。

复现使用[development](../development.md)与JSON命令，配置匹配目标CPU架构的CUDA栈。
新checkout可重新构建，不依赖旧作业目录。GPU真机资格须在实际目标硬件单独执行。
