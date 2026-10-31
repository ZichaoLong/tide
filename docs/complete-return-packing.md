# 完整返回的传输打包

`benchmark_weighted.py`与standalone `weighted-benchmark`接受显式
`--return-packing scalar|packed`，默认`scalar`保持原逐字段
`detach().cpu().clone()`。该策略独立于placement和局部算子`--packing`；
`ExecutionSpec.return_packing`、study命令、运行身份和结果均记录实际选择。
旧记录缺少此字段时仍表示`scalar`，不重写历史身份。新增执行记录包含该字段，
因而其身份hash与旧记录分别保留。结果接纳检查拒绝不同返回策略。

`packed`仅适用于eager两种Python客户端与standalone LibTorch。它从实际完整
结果收集Tensor，按原device及精确dtype分组；每组临时cat输入最多8MiB，
同步复制到CPU后按原shape解码。相同Tensor对象可只传一次，但**每次字段出现
仍独立clone**：修改一个返回字段不能改变另一字段或live continuation。
临时设备块和CPU块各至多8MiB；完整返回本身、字段索引、CPU clone、allocator
和厂商资源另计，8MiB不是总内存上限。无常驻缓存、异步传输或精度转换。

CPU、单字段、空Tensor、非连续、非strided、量化、共轭/负视图及大于8MiB的
单Tensor保留原复制路径。float payload/scoring不混dtype，bool与int64独立；
NaN字节、signed zero、超过2^54的时钟、None、空记录和原stride均不被重解释。
全部trace、控制、逐边/端口输出、状态、history、消息、pending及统计字段仍返回。
每窗live detach、独立参考、优化器与checkpoint codec保持原路径；参考始终scalar。

resident显式拒绝`packed`。其公共库已在journal/状态解码内部复制到CPU，不能将
eager外层打包称为resident收益。修改该内部解码需要另一项明确方案与资格。
计时仍包含完整返回、打包、复制、clone及最后同步；这不是最小返回模式。
磁盘恢复后另记录`exact_resume`内存阶段，allocator峰值不重置，覆盖warmup后至恢复
结束的峰值；它不改变step计时或checkpoint格式。

检查入口：`tests/test_weighted_return_packing.py`覆盖真实两窗推理/训练、独立参考、
完整记录及精确换owner恢复；literal分组与别名/None/NaN/int64/stride按字节核对。
默认CPU；设置`TIDE_RETURN_TEST_DEVICE=cuda:0`使复制见证及FP64两窗/恢复实际使用CUDA，
无设备时失败。NPU仅选择支持的literal（排除`fp64`）及`full_snapshot`，另做FP32图验收。
standalone同构建的`weighted-return-check --device cuda:0 --dtype float32`验证实际设备复制边界；
CPU测试不代替CUDA/NPU资格，各客户端分别验收。

本候选依据独立B profile的`detach_return_sync`阶段成本提出；profile阶段比例不是
普通计时的同步占比。只在相同材料、完整独立正确性和精确恢复通过后比较scalar/packed
端到端成本。若减少复制调用却没有完整窗口收益，停止扩大本候选；若需要异步、
跨窗缓存、改变返回范围或resident内部解码，则先提出新绑定/所有权与资源方案。

[固定GPU资格与有限对照](evidence/core5-cuda-inference-20261031.md)验证三客户端完整返回、独立根和小型训练/恢复；本机NPU相关检查通过但同材料端到端成本增加，默认scalar保持。
