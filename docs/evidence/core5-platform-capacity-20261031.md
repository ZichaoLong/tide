# 同模型安全容量代表点（2026-10-31）

本报告目前是CUDA4.561B单卡实际点；同模型2/3/4卡与NPU校准点待各自实际结果，不由单卡推断。
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

## 干净安装复现

```sh
# 依human-workflow安装，site指向匹配的实际库。
python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps(next(c["config"] for c in m["cases"] if c["name"]==sys.argv[2]),indent=2))' configs/matrices/cuda-capacity-calibrated.json cuda-capacity-4561m-cards-1 > selected.json
tidegraph plan selected.json --site site.json
tidegraph job start selected.json --site site.json --output-dir jobs/new-capacity
tidegraph results docs/evidence/core5-platform-capacity-20261031.json.gz --format table
```

多卡同一模型继续保持共享组同卡、全区域函数与实际位置；普通复制不认证直接P2P/原子或resident协议。下一尺度仍需逐卡、主存、两代checkpoint共同准入，不把空闲GPU显存直接相加。

归档SHA256：`8163c1d31f6adfba216ed114ca006720b6302c8cd7362ef6ecfe19a74422590d`。
