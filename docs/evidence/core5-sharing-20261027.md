# Core-5 Aggregate 精确计算复用 CPU 资格 — 2026-10-27

固定实现 `52dee1e5da119851e97fc302f71a52098aef270d`，固定上游 `0770593`。
**543 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、workload exit 0、服务 inactive/success 已核对；[命令/环境/哈希](core5-sharing-20261027.json)。

[契约及数学对应](../weighted-compute-sharing.md)规定：只在当前合法 packed 块内，
有序来源类别、载荷 Tensor 身份、实际所用参数身份和配置完全一致时，复用一次实际 Aggregate。
普通程序默认不具备能力；数值相同的独立输入根不合并。单位广播保留真实 Tensor 别名，
完整物理来源逐事件解码；各节点状态、选择、Full、逐边输出和 pending 保持独立。

Python 与独立 C++ 调度对照逐请求标量参考。四种来源绑定的双样本、多位置 FC
核对完整记录、objective 和独立 content/output 根及连接零；region/source 绑定确有
三节点共用一次 Aggregate，node/adapted 不虚报复用。同值独立输入仍各自求值，
后续真实广播允许复用。实际 computed+reused 计数等于候选数。
三步 AdamW 比较全部参数/梯度；含 pending checkpoint 换 owner 恢复及下一次更新
比较完整状态和优化器槽。来源、Aggregate、图类别与调度相关回归同批通过。

standalone C++ 独立构造两个请求，确认仅一个代表请求、两份原始来源身份，
并手算检查系数 1 与 3 的两个使用者贡献合计 4；独立同值输入和 unpacked 路径不复用。
这项局部原生证据不替代 standalone 完整图训练矩阵。

开发 tree ae014cf867a308a1aa87c5d9527f4a62c9bdcd91 的构建和同范围 **543/0** 通过；
本固定资格复核提交后的同一实现，后续 KV 投影共享留在主工作区，不属于本报告。
复现：检出固定实现，按 [development](../development.md) 构建，运行 JSON 测试列表，
`--dtype both`、新 basetemp 和当前 TIDE_BUILD_DIR，无需旧工作目录。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1；ATen/BLAS 单线程、两个 build workers。
unit tide-core5-sharing-qualify-20261027：background.slice、Nice=10、20 分钟上限；
原始输出 artifacts/jobs/core5-sharing-qualify，build manifest 已独立保存。
本报告不认证 KV base 投影复用、mixed/resident、CUDA/NPU 真机或性能。
