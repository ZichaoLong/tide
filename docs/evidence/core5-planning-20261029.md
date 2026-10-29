# 公共core-5配置和自然规模：固定CPU资格

实现源码 `8f51ba6419e94c5ded5eeb9527146992d8426ddd`，clean；数学上游0770593。
构建资源策略前置提交d841450。环境aarch64、Python3.12.13、Torch2.12.0+cpu。

公共`planning`接管原静态清单/构造；旧v1脚本和材料hash兼容。新增模型默认展开、独立执行/
测量身份、FC/HB/LH启发拓扑和自然宽度参数规模。参数FFN固定4d，没有填充参数或core-3节点数约束。
[契约、源码、测试与复现导航](../core5-study.md)。

固定源码CPU新构建完成，**336 passed、0 failed、0 skipped**，两种CPU参考dtype。
13个定向测试文件由`scripts/qualify_core5_study.py::TESTS`固定：新配置、自然规模、真实参数/共享清单、
旧计划与core-3静态拓扑兼容、分配与熵、source绑定/执行、公共库、完整工作负载训练/恢复。
这不是所有语义/平台/规模的全矩阵；本阶段没有新增设备或大型吞吐资格。

独立检查的含义：教材HB计数1408节点/22区域/3584边与公共图生成器核对静态连接；
shape公式与实际Module核对所有参数及别名；五族/混合小图前后向和时钟检查实际绑定。
分配公式和既有独立调度测试各自保留原断言；自身恢复不能替代跨调度完整结果比较。

复现（匹配环境，固定源码的独立background.slice服务内）：

```sh
python scripts/job.py --output-dir NEW_JOB -- timeout 2850   python scripts/qualify_core5_study.py --output-dir NEW_QUALIFICATION   --build-dir NEW_CPU_BUILD --jobs half
```

本机可用320CPU，实际构建160线程，build 45.532s；测试ATen/BLAS1，pytest96.56s、阶段98.722s。
服务`core5-planning-v1-cpu-20261029.service`，background.slice、Nice10、CPU16000%、Memory768GiB、
Tasks4096、总时限3000s；terminal inactive/dead、ExecMainStatus0，持久status passed/0。
无项目性能计时与构建重叠；系统共享资源，未声称独占CPU。

原始记录`artifacts/jobs/core5-planning-v1-cpu/{status.json,task.log}`及`qualification/`；
固定只读树`artifacts/worktrees/core5-planning-v1`，构建在其`build/cpu`。这些目录不是重现前置。

- core源码SHA256：`24943f60edf861e1b7cb4dd2ce5f5cbc6d523afa8709b89e41a78c31100cee27`。
- build log SHA256：`068e3f60bbc5a45d91467746d64f3812ade28b0746cd2849f520efcd8af1fc9f`。
- test log SHA256：`999d2b776efd0589d689d1debd7d781c8f70087c7e52706d9622da85347aac28`。

数值/异步/resident流资源等旧限制仍见ROADMAP；本报告不回填任何旧失败。
后续benchmark FP16/线程/设备入口修改不在本源码资格内。
