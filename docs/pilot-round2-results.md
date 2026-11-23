# 第二轮试运行结果

任务 72/72；调度状态 `completed_with_failures`；所有任务进程终止：`True`。
主训练曝光：学习 4,523,008 token，成本计算 136,291,840 token。缺测保留 unknown，恢复参考与重放不计入学习曝光。

## 阶段与资源

| 阶段 | 状态 | 秒 | 任务 | CPU/NPU 训练峰值 |
|---|---|---:|---:|---:|
| cpu-isolated | completed | 642.3 | 6 | 1/unknown |
| npu-isolated | completed | 1,096.2 | 4 | unknown/1 |
| mixed-concurrent-a | completed | 356.5 | 13 | 4/9 |
| mixed-concurrent-b | completed | 358.9 | 13 | 4/9 |
| capacity-grid | completed | 336.8 | 12 | unknown/9 |
| lr-screen | completed | 341.3 | 18 | unknown/8 |
| selected-continuation | completed_with_limits_or_failures | 164.7 | 6 | unknown/4 |

进程树采样 RSS 峰值 30.08 GiB；实际扫描间隔中位数/最大值 1.060/2.017 秒。看守方式 `sampled_process_rss_software_watchdog`，内核内存硬上限建立状态 `False`。

## 配对并发观察

比值为相同测量前缀的并发吞吐/独立吞吐；计划负载不足仍明确列出实际观察范围。

| 组 / 并发任务 | 共同/测量步 | 比值 | 训练并发峰值 | 计划负载覆盖 | 资格 |
|---|---:|---:|---:|---:|---|
| cpu-exchange-x-t16 / unknown | unknown/unknown | unknown | unknown | unknown% | isolated_only_design; unknown |
| cpu-exchange-x-t4 / cpu-concurrent-exchange-x-a | 65/62 | 0.986 | 13 | 29.2% | matched_prefix; insufficient_or_variable_training_concurrency |
| cpu-exchange-x-t4 / cpu-concurrent-exchange-x-b | 65/62 | 0.983 | 13 | 27.2% | matched_prefix; insufficient_or_variable_training_concurrency |
| cpu-exchange-x-t40 / unknown | unknown/unknown | unknown | unknown | unknown% | isolated_only_design; unknown |
| cpu-transformer-t16 / unknown | unknown/unknown | unknown | unknown | unknown% | isolated_only_design; unknown |
| cpu-transformer-t4 / cpu-concurrent-transformer-a | 1,391/1,388 | 0.989 | 13 | 25.3% | matched_prefix; insufficient_or_variable_training_concurrency |
| cpu-transformer-t4 / cpu-concurrent-transformer-b | 1,391/1,388 | 0.980 | 13 | 25.9% | matched_prefix; insufficient_or_variable_training_concurrency |
| cpu-transformer-t40 / unknown | unknown/unknown | unknown | unknown | unknown% | isolated_only_design; unknown |
| npu-exchange-x-a / npu-concurrent-exchange-x-a | 338/335 | 0.974 | 13 | 61.9% | matched_prefix; insufficient_or_variable_training_concurrency |
| npu-exchange-x-b / npu-concurrent-exchange-x-b | 341/338 | 0.984 | 13 | 62.3% | matched_prefix; insufficient_or_variable_training_concurrency |
| npu-transformer-a / npu-concurrent-transformer-a | 8,105/8,102 | 0.967 | 13 | 65.0% | matched_prefix; insufficient_or_variable_training_concurrency |
| npu-transformer-b / npu-concurrent-transformer-b | 8,137/8,134 | 0.968 | 13 | 65.6% | matched_prefix; insufficient_or_variable_training_concurrency |

## 学习率筛选与接续

声明父任务 18 个、接续任务 6 个；观察到 selection.json 4 份。

| 模型 | LR | 种子 | 全种子平均 NLL | 全部完成 / 严格 8 步 |
|---|---:|---|---:|---|
| fc | 0.0001 | [20261119, 20261120] | 7.40061 | True / True |
| fc | 0.0003 | [20261119, 20261120] | 6.99036 | True / True |
| fc | 0.001 | [20261119, 20261120] | 6.78476 | True / False |
| tf | 0.0001 | [20261119, 20261120] | 6.95312 | True / True |
| tf | 0.0003 | [20261119, 20261120] | 6.72128 | True / True |
| tf | 0.001 | [20261119, 20261120] | 6.73744 | True / True |
| x | 0.0001 | [20261119, 20261120] | 8.09414 | True / False |
| x | 0.0003 | [20261119, 20261120] | 7.25367 | True / False |
| x | 0.001 | [20261119, 20261120] | 6.92341 | True / False |

| 接续任务 | 父任务 | 获选 LR / 种子均值 NLL | 全局范围 | 新增步 | 严格 8 步 |
|---|---|---:|---|---:|---|
| continue-fc-s20261119 | screen-fc-lr0.0003-s20261119 | 0.00030 / 6.99036 | 257→1025 | 768 | True |
| continue-fc-s20261120 | screen-fc-lr0.0003-s20261120 | 0.00030 / 6.99036 | 257→1025 | 768 | True |
| continue-x-s20261119 | unknown | unknown / unknown | unknown→unknown | unknown | False |
| continue-x-s20261120 | unknown | unknown / unknown | unknown→unknown | unknown | False |
| continue-tf-s20261119 | screen-tf-lr0.0003-s20261119 | 0.00030 / 6.72128 | 257→1025 | 768 | True |
| continue-tf-s20261120 | screen-tf-lr0.0003-s20261120 | 0.00030 / 6.72128 | 257→1025 | 768 | True |

## 每任务记录

| 任务 | 用途 | 状态 | token/s | worker 秒 | 段步 / 全局终点 | RSS GiB / NPU GiB | 恢复步数 / 严格 |
|---|---|---|---:|---:|---:|---:|---|
| cpu-isolated-exchange-x-t4 | cost | completed | 367.3 | 111.2 | 65/65 | 3.46/unknown | 1/True |
| cpu-isolated-exchange-x-t16 | cost | completed | 450.7 | 111.7 | 81/81 | 4.08/unknown | 1/True |
| cpu-isolated-exchange-x-t40 | cost | completed | 394.4 | 112.1 | 70/70 | 4.32/unknown | 1/True |
| cpu-isolated-transformer-t4 | cost | completed | 8,130.0 | 98.7 | 1391/1391 | 2.53/unknown | 1/True |
| cpu-isolated-transformer-t16 | cost | completed | 13,698.6 | 99.3 | 2297/2297 | 2.99/unknown | 1/True |
| cpu-isolated-transformer-t40 | cost | completed | 13,700.9 | 98.6 | 2295/2295 | 2.82/unknown | 1/True |
| npu-isolated-exchange-x-a | cost | completed | 2,976.4 | 273.1 | 347/347 | 5.50/4.11 | 1/True |
| npu-isolated-transformer-a | cost | completed | 76,196.7 | 254.8 | 8397/8397 | 5.21/0.75 | 1/True |
| npu-isolated-exchange-x-b | cost | completed | 2,968.3 | 272.7 | 346/346 | 5.49/4.11 | 1/True |
| npu-isolated-transformer-b | cost | completed | 76,397.3 | 255.0 | 8409/8409 | 5.21/0.75 | 1/True |
| npu-concurrent-exchange-x-a | cost | completed | 2,900.3 | 273.0 | 338/338 | 5.49/4.11 | 1/True |
| npu-concurrent-transformer-a | cost | completed | 73,687.6 | 255.5 | 8105/8105 | 5.21/0.75 | 1/True |
| npu-load-fc32-a | cost | completed | 13,067.2 | 262.2 | 1511/1511 | 6.44/2.91 | 1/True |
| npu-load-x32-a | cost | completed | 2,971.3 | 274.6 | 346/346 | 5.96/4.33 | 1/True |
| npu-load-x64-a | cost | completed | 4,767.2 | 278.7 | 278/278 | 7.29/8.42 | 1/True |
| npu-load-x16-l128-a | cost | completed | 2,901.6 | 273.0 | 338/338 | 5.49/4.11 | 1/True |
| npu-load-x16-l256-a | cost | completed | 4,084.6 | 275.1 | 238/238 | 5.87/7.92 | 1/True |
| npu-load-timed16-a | cost | completed | 692.9 | 289.9 | 162/162 | 7.04/11.75 | 1/True |
| npu-load-shannon8-a | learning | completed | 839.3 | 292.0 | 390/390 | 5.00/1.21 | 8/True |
| cpu-concurrent-exchange-x-a | cost | completed | 362.0 | 264.1 | 170/170 | 3.99/unknown | 1/True |
| cpu-concurrent-transformer-a | cost | completed | 8,039.5 | 248.8 | 3665/3665 | 2.61/unknown | 1/True |
| cpu-load-exchange-x-a | cost | completed | 452.0 | 263.7 | 213/213 | 3.98/unknown | 1/True |
| cpu-load-transformer-a | cost | completed | 10,662.5 | 249.1 | 4791/4791 | 2.79/unknown | 1/True |
| npu-concurrent-exchange-x-b | cost | completed | 2,921.9 | 273.1 | 341/341 | 5.50/4.11 | 1/True |
| npu-concurrent-transformer-b | cost | completed | 73,974.4 | 254.9 | 8137/8137 | 5.21/0.75 | 1/True |
| npu-load-fc64-b | cost | completed | 19,309.5 | 265.8 | 1121/1121 | 7.27/5.66 | 1/True |
| npu-load-x32-l128-b | cost | completed | 4,547.2 | 278.2 | 265/265 | 6.33/8.15 | 1/True |
| npu-load-x64-l128-b | cost | completed | 6,041.0 | 285.7 | 177/177 | 7.51/15.94 | 1/True |
| npu-load-x16-l256-b | cost | completed | 4,138.9 | 277.7 | 241/241 | 5.87/7.92 | 1/True |
| npu-load-timed-mixed16-b | cost | completed | 777.3 | 278.3 | 181/181 | 6.63/11.53 | 1/True |
| npu-load-timed16-l128-b | cost | completed | 779.3 | 296.8 | 91/91 | 7.20/22.28 | 1/True |
| npu-load-shannon16-b | learning | completed | 1,600.6 | 294.3 | 373/373 | 5.35/2.33 | 8/True |
| cpu-concurrent-exchange-x-b | cost | completed | 360.1 | 263.1 | 169/169 | 4.09/unknown | 1/True |
| cpu-concurrent-transformer-b | cost | completed | 7,966.2 | 249.2 | 3633/3633 | 2.91/unknown | 1/True |
| cpu-load-exchange-x-b | cost | completed | 437.9 | 261.5 | 206/206 | 3.46/unknown | 1/True |
| cpu-load-transformer-b | cost | completed | 11,392.6 | 248.9 | 5115/5115 | 2.87/unknown | 1/True |
| capacity-x8-l64 | cost | completed | 925.1 | 150.6 | 215/215 | 4.92/1.15 | 1/True |
| capacity-x16-l64 | cost | completed | 1,759.6 | 152.5 | 205/205 | 5.22/2.21 | 1/True |
| capacity-x32-l64 | cost | completed | 3,113.4 | 153.2 | 181/181 | 5.75/4.32 | 1/True |
| capacity-x64-l64 | cost | completed | 4,896.2 | 157.4 | 143/143 | 7.23/8.41 | 1/True |
| capacity-x8-l128 | cost | completed | 1,645.0 | 153.2 | 193/193 | 5.12/2.13 | 1/True |
| capacity-x8-l256 | cost | completed | 2,426.3 | 153.1 | 141/141 | 5.22/4.02 | 1/True |
| capacity-fc8-l64 | cost | completed | 3,639.2 | 141.0 | 841/841 | 5.32/0.91 | 1/True |
| capacity-fc16-l64 | cost | completed | 7,331.8 | 140.1 | 841/841 | 5.71/1.56 | 1/True |
| capacity-fc32-l64 | cost | completed | 13,180.6 | 141.6 | 757/757 | 6.22/2.91 | 1/True |
| capacity-tf8-l64 | cost | completed | 20,064.3 | 133.5 | 4370/4370 | 4.84/0.30 | 1/True |
| capacity-tf32-l128 | cost | completed | 119,391.6 | 135.7 | 3326/3326 | 5.72/1.42 | 1/True |
| capacity-tf64-l128 | cost | completed | 152,484.7 | 136.4 | 2161/2161 | 5.81/2.74 | 1/True |
| screen-fc-lr0.0001-s20261119 | learning | completed | 3,693.8 | 71.6 | 257/257 | 5.46/0.93 | 8/True |
| screen-fc-lr0.0001-s20261120 | learning | completed | 3,762.1 | 76.0 | 257/257 | 5.44/0.91 | 8/True |
| screen-fc-lr0.0003-s20261119 | learning | completed | 3,747.9 | 77.4 | 257/257 | 5.47/0.92 | 8/True |
| screen-fc-lr0.0003-s20261120 | learning | completed | 3,717.3 | 77.0 | 257/257 | 5.46/0.92 | 8/True |
| screen-fc-lr0.001-s20261119 | learning | completed | 3,638.2 | 77.2 | 257/257 | 5.47/0.90 | 8/False |
| screen-fc-lr0.001-s20261120 | learning | completed | 3,689.2 | 79.5 | 257/257 | 5.46/0.91 | 8/False |
| continue-fc-s20261119 | learning | completed | 3,787.5 | 145.2 | 768/1025 | 5.66/0.79 | 8/True |
| continue-fc-s20261120 | learning | completed | 3,734.9 | 146.4 | 768/1025 | 5.68/0.83 | 8/True |
| screen-x-lr0.0001-s20261119 | learning | completed | 912.0 | 202.3 | 257/257 | 5.02/1.17 | 8/True |
| screen-x-lr0.0001-s20261120 | learning | completed | 908.4 | 202.9 | 257/257 | 5.02/1.17 | 8/False |
| screen-x-lr0.0003-s20261119 | learning | completed | 897.4 | 205.7 | 257/257 | 5.03/1.17 | 8/False |
| screen-x-lr0.0003-s20261120 | learning | completed | 912.3 | 200.4 | 257/257 | 5.03/1.16 | 8/False |
| screen-x-lr0.001-s20261119 | learning | completed | 841.3 | 219.7 | 257/257 | 5.02/1.16 | 8/False |
| screen-x-lr0.001-s20261120 | learning | completed | 825.8 | 222.8 | 257/257 | 5.01/1.15 | 8/False |
| continue-x-s20261119 | learning | failed | unknown | unknown | unknown/unknown | unknown/unknown | unknown/False |
| continue-x-s20261120 | learning | failed | unknown | unknown | unknown/unknown | unknown/unknown | unknown/False |
| screen-tf-lr0.0001-s20261119 | learning | completed | 19,848.4 | 27.9 | 257/257 | 4.79/0.30 | 8/True |
| screen-tf-lr0.0001-s20261120 | learning | completed | 18,978.5 | 30.6 | 257/257 | 4.79/0.30 | 8/True |
| screen-tf-lr0.0003-s20261119 | learning | completed | 19,839.5 | 28.4 | 257/257 | 4.79/0.30 | 8/True |
| screen-tf-lr0.0003-s20261120 | learning | completed | 19,650.3 | 28.1 | 257/257 | 4.79/0.30 | 8/True |
| screen-tf-lr0.001-s20261119 | learning | completed | 19,900.3 | 27.7 | 257/257 | 4.79/0.30 | 8/True |
| screen-tf-lr0.001-s20261120 | learning | completed | 20,217.7 | 28.2 | 257/257 | 4.79/0.30 | 8/True |
| continue-tf-s20261119 | learning | completed | 20,446.9 | 44.8 | 768/1025 | 4.95/0.30 | 8/True |
| continue-tf-s20261120 | learning | completed | 20,798.6 | 44.2 | 768/1025 | 4.95/0.30 | 8/True |

## 解释范围

- 第二轮逐步训练计时包含 input/RNG 指纹；第一轮测量边界不同，跨轮吞吐不作为纯性能回归或架构排名。
- 配对比值只使用相同模型、配方、源码、CPU mask（NPU 还须相同 physical card）且输入/RNG 完全一致的共同 step 前缀，并去掉共同 warmup。
- 配对固定声明工作负载与公共输入/RNG，未逐更新捕获或验证路由及全部活动张量相等；冷态活动探针不补足此证据，比值不称为严格实现等价加速。
- 训练区间由 steps.jsonl 的 time_unix 和 seconds 重建，终点为记录时间，存在记录开销偏移；交集仅覆盖主训练更新，不覆盖验证、恢复和未知外部负载。
- worker 同时存活与训练更新重叠分别统计；实际并发不足时不声称达到计划并发条件。
- lr=0 成本任务执行前向、反向、裁剪和 AdamW；其参数不变符合设计，token 不计为学习证据。
- 学习 token 只计保存轨迹中的主训练更新；同 owner 参考更新和新进程恢复验证不增加该 checkpoint 的学习曝光。
- 18 个父任务的三档学习率与两个种子仅用于模型内探索筛选；固定文章留出探针不能建立架构质量排名。
- 恢复逐步比较保留 atol=1e-5、rtol=1e-4，整数和结构严格相等；梯度记录位于全局范数裁剪后。
- 因果探针在每段开始时使用空私有状态、B=1、16 token；接续段使用恢复后的权重，不认证全部训练状态和轨迹。
- RSS 是进程树软件看守的离散采样，共享页可能重复计入，采样间可超过阈值；不等于内核硬上限，也不包含文件缓存或 NPU 设备内存。
- 退出状态、浮点、离散、恢复、学习和性能分别报告；缺失或失败的测量不会填成零。

JSON 包含每步恢复判据、输入身份、解析记录、选择复算及逐任务测量；长日志只保留原路径、SHA256 和汇总证据。
