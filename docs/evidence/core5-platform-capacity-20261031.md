# 同模型安全容量代表点（2026-10-31）

本报告包括CUDA4.561B同模型单/双/三/四卡实际点；NPU校准点待各自实际结果，不由已测点推断。
固定Python实现bd2ff2e，core/resident构建de999c2；配置31afefb，实际冻结输入由ad93848导出。
RTX3090/sm86、Torch2.12.1+cu130、FP32 payload/scoring、TF32关闭，native mixed-c、ordinary复制、packed/scalar完整返回；ATen/interop/worker1。
FC16/8区域、KV、d4352、FFN4d，4,561,319,928个去重可训练参数，8个不可拆共享组；B1/P2/span20，staggered-v1，warmup1+连续2窗+精确恢复窗，1800秒上限。

## 单卡实际准入与结果

实际参数+冻结bank19,458,826,704 bytes；逐卡另预留1GiB workspace、256MiB activation、64MiB通信及3GiB不用。
静态峰值20,941,417,120 bytes；现场扣除3GiB后的usable21,798,912,000 bytes，预算剩余857,494,880 bytes。
主存预算91,660,099,776 bytes；两代checkpoint磁盘预算42,482,056,608 bytes，另留1GiB空盘。
先审阅并退役28.505GB终态可再生checkpoint，保留结果/配置/log，再以live free43,976,364,032 bytes准入。不是反复OOM搜索，也没有缩小batch/cache或改变精度。

| 结果 | 实际值 |
| --- | --- |
| 执行 / 独立数值 / 离散 / 精确恢复 | 完成 / 超差探索 / 通过 / 通过 |
| 两个测量窗 / 中位 | 0.595189 / 0.755775秒；中位0.675482秒 |
| 每窗candidate / Full / 消息 | 31 / 18和16 / 32和28 |
| 构造 / 独立参考 / 恢复检查 | 36.822 / 41.689 / 314.967秒 |
| 全程allocator peak allocated / reserved | 20,058,310,144 / 20,245,905,408 bytes |
| RSS高水位 | 74,476,953,600 bytes |
| 最后checkpoint单代 | 20,699,246,016 bytes；next_window3 |

allocated峰值约占扣除余量后usable的92.0%；reserved约92.9%，属于该口径下的单卡近界代表。allocator不含驱动/厂商隐藏内存；不称硬件绝对最大模型。
逐节点参数、实际state/message/pending及计算统计与请求单卡owner映射相符。完整返回、状态/控制/逐边端口/history/pending比较与磁盘下一窗各自记录。
原数值首差window0/sample0/node0 KV keys，133/4352坐标超原atol1e-6/rtol1e-5，最大2.81334e-5；六份比较累计1512数值差异、0离散差异。不能称严格合格吞吐。
完整checkpoint恢复远大于单窗计算成本，实验选型须同时考虑持久化和恢复；0.675秒不能代表整个资格流程耗时。

任务`jobs/platform-cuda-capacity4561m-c1-bd2ff2e`于06:53:40Z exit0，独立SSH检查supervisor/workload/remaining全部为空。
保存配置、结果、来源与hash后才退役可重建checkpoint，为同模型下一点释放盘；活动输入不退役。

## 同模型双卡实际点

双卡沿用完全相同model_id/workload_id、B1/P2/span20与完整返回，8共享组交替放在GPU0/1，每卡8节点。
执行完成、离散语义和磁盘精确续接通过；1517数值差异、0离散差异，首差仍为原keys坐标，保持FP32探索。

| 指标 | 双卡实际值 |
| --- | --- |
| 两测量窗 / 中位 | 0.683779 / 0.825447秒；0.754613秒 |
| 每卡峰值 allocated / reserved | 10.299098 / 10.441720GB；两卡近似相同 |
| 主存RSS峰值 / 恢复检查 | 74.808754GB / 320.717秒 |
| 原生跨卡复制 | 各窗7组；557184 / 487536 bytes；ordinary生效 |
| 实际位置 | 参数、私有状态、消息目标和计算与共享组映射相符；无参数独立复制替代共享 |

候选/Full/消息与单卡相同，31/(18,16)/(32,28)。相同模型双卡降低逐卡存储但本次单窗成本高于单卡；
不将两卡空闲显存之和称容量，也不由ordinary复制推断直接P2P或resident协议。
任务`jobs/platform-cuda-capacity4561m-c2-bd2ff2e`于07:11:16Z exit0，独立SSH核对supervisor/workload/remaining全空。
结果原字节SHA256 `3bb5e0ac7fe851677fd47329472fd31aeb656de1cb96c829a84384b6089054d6`。

## 同模型三卡实际点

同一模型/工作负载/任务与完整返回，三卡6/6/4节点；执行、离散、精确恢复通过，1517数值差异/0离散差异保持探索。
两测量窗0.719262/0.839067秒、中位0.779165秒；candidate/Full/消息与前两点相同。
allocated峰值7.859294/7.859295/5.419492GB，reserved7.990149/7.992246/5.540676GB；RSS74.883752GB。
构造40.413秒、独立参考42.921秒、精确恢复316.800秒；原生跨卡两窗仍各7组、557184/487536 bytes。
逐节点实际state、message目标、pending与有效设备映射核对相符；共享组未拆成独立参数。
任务`jobs/platform-cuda-capacity4561m-c3-bd2ff2e`于07:23:53Z exit0，SSH检查supervisor/workload/remaining全空。
结果原字节SHA256 `19a099693a6d09634fb8e65c53f6dee17843759b8da010e4e49bf878f384334d`。

## 干净安装复现

```sh
# 依human-workflow安装，site指向匹配的实际库。
python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps(next(c["config"] for c in m["cases"] if c["name"]==sys.argv[2]),indent=2))' configs/matrices/cuda-capacity-calibrated.json cuda-capacity-4561m-cards-1 > selected.json
tidegraph plan selected.json --site site.json
tidegraph job start selected.json --site site.json --output-dir jobs/new-capacity
tidegraph results docs/evidence/core5-platform-capacity-20261031.json.gz --format table
```

多卡同一模型继续保持共享组同卡、全区域函数与实际位置；普通复制不认证直接P2P/原子或resident协议。下一尺度仍需逐卡、主存、两代checkpoint共同准入，不把空闲GPU显存直接相加。

## 同模型四卡实际点

同一4.561B模型/任务/输入与完整返回，四卡各4节点，实际参数/私有state/消息目标/pending与放置映射核对相符。
两测量窗0.721320/0.847173秒，中位0.784246秒；每窗31 candidate，Full18/16、消息32/28。
执行、完整离散检查和精确下一窗恢复通过；1517数值差异、0离散差异，首差仍为原keys，保持FP32探索。
四卡allocated峰值5.419491/5.419492/5.419492/5.419491GB，reserved各5.540676GB，RSS74.944528GB。
构造39.900秒、独立参考42.857秒、精确恢复281.042秒；复制每窗7组、557184/487536 bytes，ordinary生效。
同模型1→4卡降低逐卡存储，本次成本0.675→0.784秒，没有速度优势；不是四卡总容量极限。
任务07:34:04Z终态exit0，独立SSH检查无supervisor/workload/remaining后代。
结果原字节SHA256 `1794416ad9ce21ed4b60a8a22fe4bd90f1b88da6509684ad8ca5dcbfc2093ba3`。
本次提交后按用户要求暂停新增工作，四卡checkpoint仍保留；9.123B现场准入尚未执行，不由静态预算写成现场拒绝。

归档SHA256：`6a0d1271e0ba66b44ed4648442f5da61e7675b3aaac8ac3b1c1d2112127e7e1c`。
