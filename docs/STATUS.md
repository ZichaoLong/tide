# 当前交接 — 2026-10-31

当前任务：core-5 CUDA 真机开发、资格与有限选型；授权分支 `graph-execution-foundation`，允许阶段提交、普通推送并继续。
数学固定 `0770593838a37c790a568226930567ed60f99b83`；19份core-5和旧core-3快照校验通过。
启动实现 `a066756fe23260e45d61afa158f87883d426f0d5`；接续HEAD `18cafd7`，已有源码不回退。
历史来源/新分支政策见 [history-migration](history-migration.md)，只读参考不变，无子代理。

## 已交付和严格边界

实现提交 `cc6b45f`（持久容器作业）、`c962961`（eager完整训练/精确恢复）、`7fd0576`（明确单卡resident库存）、
`1fe1e37`（CUDA13 API）、`31d0afd`（IEEE/P2P根探针）、`c368b45`（独立失败继续）、
`6cacad0`（稳定字段packing/standalone根）、`8a34f66`（CUDA所有权/增量预算）、`32edb0f`（JSON头/IEEE）均已提交推送。
有限目录 `a864fef`，前两阶段审阅证据 `e648027`、`18cafd7`。

- [eager证据](evidence/core5-cuda-eager-20261031.md)：三客户端构建/加载实际执行；单卡PyTorch和native两dtype各19/19，standalone FP32 23/24、FP16 22/24；KV mixed-a/b三客户端6/6。
- [单卡resident及补验](evidence/core5-cuda-resident-20261031.md)：固定 `32edb0f2b035ef2b732b6f27cf577f9e98058eb6` 四阶段新构建通过，56线程/SM86；单卡56组件/dtype+53公共项零跳过通过。
  新standalone独立根随原矩阵验收；scalar FP32六代表×三客户端18/18，standalone双卡六代表6/6，native FP16六代表6/6；CPU相关110项和C++两dtype局部核通过。
- 原standalone FP32 fc/ssd第三步AdamW bias差3.4869e-6、mixed-a精确恢复、FP16 feedback/affine及feedback/ssd、native双卡GDN精确恢复失败均保留。
  两项FP16在CPU复现同值，SSD显式scalar CPU/CUDA通过，affine scalar仍失败；不能称packed修复或CUDA专有错误。
- 普通复制72单元、standalone通用双卡owner36例/72更新/144连接窗通过；显式Python CPU桥delta-pdg完整通过。
  所有12方向直接P2P/原子不支持、enable217；不认证直接P2P/NCCL/多卡resident。AccumulateGrad流警告保留。
- 完整三卡门禁与原严格断言/xfail保持；旧CPU/NPU边界及本轮未解除限制只见 [ROADMAP](ROADMAP.md)。

## 目标机、来源与作业

通过 `artifacts/cuda3090/ssh.sh` 使用当前认证CONNECT代理连接 `gpu3090`；凭据未写入证据，无全局配置修改。
目标根 `/root/autodl-tmp/gef-cuda3090`；项目私有 `venv` 为Python3.12.3/Torch2.12.1+cu130，pytest8.4.2/Ninja1.13.0，
nlohmann_json3.11.2；Toolkit13.0.88/GCC11.4/CMake3.22.1/Nsight2025.3.1，四张RTX3090/24GiB/SM86。
CPU affinity112、配额56核，内存240GiB。接续检查数据盘剩48.6GiB、cgroup内存约4.3GiB；GPU0在运行有限对照，其他三卡空闲。
冻结源码 `sources/32edb0f` 只读，匹配 `builds/32edb0f-{core,resident-cpp,resident-python,benchmark}`；不得编辑活跃作业输入。
`dev` 是独立dirty开发目录，不能作为资格来源。本机无长作业。

所有目标任务用容器subreaper监督器；SSH断连继续与stop后无后代已实际验证。
每个作业日志 `jobs/NAME/task.log`、状态 `jobs/NAME/status.json`；完整输出 `runs/NAME`，旧失败不覆盖。
已终态：build-all-32edb0f passed、resident-single-32edb0f passed、eager-regression-32edb0f failed、
profiles-32edb0f passed、half-boundary-32edb0f failed、half-packing-32edb0f failed，全部无存活后代。
早期86-real/API构建及resident-inventory-c368b45失败保持原身份，后续通过不能重标旧结果。

## 当前工作与下一动作

`jobs/finite-32edb0f` running，物理GPU0、10800秒总上限，每项780秒；37个显式选择串行，1warmup+3测量窗口。
输出 `runs/32edb0f-finite`；冻结实现32edb0f，目录配置a864fef，最大约345万参数。
已观察小型FC八组合、小型HB eager通过；中型FC七组合均严格失败（含scalar），HB小型resident两客户端在同一values坐标失败。
完整状态、最终梯度/SGD槽及同栈精确恢复全部通过才授予吞吐；仍运行的项无通过结论。

profile作业已终态：control与两个完整resident消费者Nsight node trace、两种eager客户端Torch profile。
resident消费者profile内完整核对/精确恢复通过；不是吞吐样本。
当前新增 `scripts/inspect_cuda_timeline.py`：只读SQLite、进程/设备域、kernel覆盖并集、host launch分列，
保留缺失correlation且不猜launch归属；7项独立SQLite见证测试通过。
下一步先提交observer，再用固定observer分析已终态trace；收齐finite终态、逐配置严格失败与可用时间，单独提交审阅证据并普通推送。
不得扩展为大规模扫描或重复相同失败候选；依据实际profile决定有无值得实施的新优化。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/venv/bin/python /root/autodl-tmp/gef-cuda3090/sources/32edb0f/scripts/container_job.py status --output-dir /root/autodl-tmp/gef-cuda3090/jobs/finite-32edb0f'
python -m unittest discover -s tests -p test_cuda_timeline.py -v
```

无旧运行目录的重建/资格命令见上述两份CUDA报告与 [development](development.md)；
有限配置入口 `benchmarks/core5-cuda-finite-v1.json` / `scripts/run_core5_study.py`。
新代码和证据分开提交；未提交文件以 `git status` 为准。
