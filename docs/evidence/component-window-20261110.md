# 默认分量窗口的固定资格（2026-11-10）

实现 `da2f41b70a442f6ef584743b057fd0859acc20de`；测试修正 `b148ba1`。
数学固定0770593。[机器记录](component-window-20261110.json)保留逐组数量、严格失败和作业来源。

默认host prefill按静态区域SCC分解，在完整外部输入封闭窗内在线求值。
无内部消息环的分量处理整窗，循环分量使用原在线关闭证书；一般PDG、非对齐DAG、
一般Settle端口仍接受。局部能力/历史/Next限制保留后备；不能推断最少kernel或最优时间。
Python与C++独立调度，standalone入口同样接线；resident不由host策略冒充。

固定安装CPU四组共391项通过。NPU五记忆×HB/非对齐DAG/反馈PDG×两个客户端：
28通过，2个GDN/HB evidence独立根严格超差，最大约3.8892e-6；原阈值1e-6/1e-5保持。
两例的完整记录、离散支持、两步梯度/SGD与同设备逐位恢复均完成；超差原因待有限诊断，
不能因有限值直接认定无害舍入。另有NPU密集路径33项相关检查通过。

Python/native安装消费者各三步独立训练及新进程精确恢复通过。
独立SDK的CPU FP64、CPU FP32、NPU FP32三个消费者均通过独立训练与新进程精确恢复。
SDK构建使用独立LibTorch-NPU 2.12/CANN9.2，不链接Python wheel运行库。
构建默认320核，无本项目重型性能计时重叠。

初始开发构建缺头文件、旧PDG拒绝断言、启动器变量替换错误和跨设备测试器误拒绝均保留
原失败。测试器修正只移除跨CPU/NPU比较的同设备要求；精确恢复仍在同设备逐位比较。
不将后继作业通过改写为先前失败作业通过。

复现：固定提交安装wheel/构建后，运行`test_component_window*.py`、`test_greedy.py`、
`test_settle_ports.py`、`test_region_contract.py`与`test_region_schedules.py`（路径前缀`tests/`）；
NPU设置`TIDE_WINDOW_DEVICE=npu:0`及`TIDE_DENSE_NUMERIC_REPORT=NEW`，保留超差报告。
安装消费者使用`qualify_dense_query.sh NEW npu MATCHING_BUILD auto`；
standalone使用已有weighted benchmark导出/消费者入口。精确运行argv在机器记录和各来源脚本。

本报告不包括原FC16性能、大宽度/大量节点、FP16、多卡、resident或CUDA新资格。
