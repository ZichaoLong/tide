当前局部 grouped FFN 设备证据（源码05e8338b）

reference/shared 的独立 trace 在 Ascend Hardware/stream515 记录27个 GroupedMatmul，Task Type均为KERNEL_MIX_AIC；同负载另一趟dispatch为18次group0、9次group2。关键GMM设备执行已有实际任务轨见证。

分组还执行2次SortAiCpu（合计82.880微秒）和2次CumsumAiCpu（66.384微秒），均为设备上的KERNEL_AICPU，不是host CPU执行。上述诊断时间不能当正常wall份额。

所有group0权重输入都别名原Parameter库/转置视图；四case grouped的ATen参数gather为0，未观察到ATen参数layout copy。stderr仍有32-padding需要分配内存的CANN/SoC警告，trace中的9次SDMA和3次TensorMove不带参数归属，不能称vendor内部零分配/零搬运。

原四项d768输出超2e-6/2e-6阈值仍保留，两条NPU路径都超CPU独立FP32公式；同设备gather与grouped的直接差异须另行固定输入诊断。

对应JSON：artifacts/grouped-ffn-development/trace-review.json；原记录及trace的精确SHA在该JSON中。
