# 当前交接 — 2026-10-28

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 `graph-execution-foundation`。
最近已推送a2fb5bd（样本副本及master复验证据）；本提交归档有限性能并修复公共导航中的过时采用/能力描述。
用户授权持续实施、有限实验、分阶段提交与正常推送；不用子代理。总体goal仍active，**未完成**。
唯一待办 [ROADMAP](ROADMAP.md)，持续规则 [execution-flows](execution-flows.md)。参考仓库/公共环境只读。

## 已实现与已验证

- core-5公共表示、同秩一般Settle/端口、五记忆及正式发送/评分/证据/key、独立调度、
  packed/VJP、mixed、常驻FP32/FP16图/完整一阶训练及续接证据从 [semantic-map](semantic-map.md) 导航。
- [21ed86d完整FP16图/训练](evidence/core5-resident-half-training-20261028.md)：CPU32、Python-owned NPU52，
  standalone两dtype各21图/15配置/20更新；CUDA三架构构建，GPU真机待验。
- [7db0b08样本副本与master复验](evidence/core5-resident-weighted-peers-20261028.md)：
  standalone两dtype各20训练更新（修正后的master断言）、6次双卡共同更新通过；CPU guard4。
  Python-owned672e948四项/62.496秒通过，CUDA264步及68条选定三架构CUDA编译通过。
  两批NPU物理9/13→0/1；三个unit终态与source/binary/loader/log/XML hash均审阅。
  原672e948 standalone bad_alloc失败及21ed86d悬空master断言审计补注保留，不改写旧状态。
- 样本副本拥有各卡完整图及私有状态；**单样本跨卡节点/状态切分尚未实现**，不能由上述资格外推。

## 有限性能已终态、已审阅

[性能报告](evidence/core5-weighted-performance-20261028.md)及同名JSON：固定脚本/fixture
`ceae7bf3173cc3c85250ff6ed46a6470b77a6971`。24配置22通过；CPU反馈PDG两项严格浮点失败。
每配置1个进程、1warmup+3连续测量窗，FP32/SGD、每窗detach，全部计入调度/packing/完整CPU返回/反向/optimizer/同步。
CPU/Python-owned Torch2.12；NPU Python CANN9.1，standalone SDK2.10/CANN9.0；各自身份独立。

| job | 终态与范围 |
| --- | --- |
| core5-weighted-performance-cpu | failed/exit1；13配置完整对照通过、PDG streaming/prefill两项严格失败 |
| core5-weighted-performance-npu-python | passed/exit0；resident small/medium训练与medium推理、Python/native mixed-c各一项，共5项 |
| core5-weighted-performance-npu-standalone | passed/exit0；KV small/medium训练、medium推理及SSD medium训练，共4项 |
| core5-weighted-medium-profiles | passed/exit0；KV/HB medium resident与mixed-c完整周期profile，各自事后完整对照 |
| core5-weighted-benchmark-{cpu,npu}-entry | passed/exit0；新消费者构建/loader/help及CPU三项、NPU一项入口，不作正式吞吐 |

NPU性能两组均物理9→npu:0，本项目性能任务串行。CPU期间另有NPU正确性资格，争用记录保留。
CPU HB/KV medium standalone prefill约350ms、streaming约456ms；NPU mixed-c约1.41–1.61秒、resident约2.60–2.71秒。
这是有限尺寸的实际选择，非统计稳定/普遍最快结论；没有稀疏发送、多卡或CUDA吞吐结论。
构造/warmup单列；runtime/loader冷启动没有独立计时，SDK allocator峰值未导出，需补有限指标。

冻结checkout：`artifacts/worktrees/core5-weighted-measurement`（ceae7bf，跟踪文件只读）。
CPU consumer另在`artifacts/worktrees/core5-weighted-benchmark/build/cpu`（3516bef，消费者C++源码未变）；
NPU consumer在上述measurement的`build/npu`，匹配7db resident。各component/core/binary身份已审阅。
静态fixture在`artifacts/core5-weighted-measurement-fixtures/`，5个配置，只含静态参数/输入/别名，无轨迹。

units均`tide-JOB-20261028.service`，输出`artifacts/jobs/JOB/`（status/task.log/measurements/各case result及log）。
launcher `.local/JOB.sh`；只读清单`.local/core5-weighted-performance-cases.json`与harness`.local/run-core5-weighted-performance.py`。
审阅脚本`.local/record-weighted-performance.py`，peer审阅`.local/record-weighted-peer-evidence.py`。
CPU每项120秒/组1500秒，NPU每项180秒/组960秒/queue180秒；Nice10/background.slice、ATen/BLAS1。
**当前没有活动任务。**

## 新严格失败与必要保留输入

FP32 PDG两项初次报maxabs4.11272e-6与2.71201e-6（shape33），不能作等价加速结论。
有限诊断`core5-weighted-pdg-rounding-diagnose`完成exit0：独立Python与两native日程、四窗口SGD，
两dtype离散结构相同；FP32首先在pending/后在raw出现差异，最终参数仍过原容差；
同一FP32初值提升FP64后完整记录/参数全部通过。此诊断不改变原失败，也不是新的性能测量。
保留CPU测量job中的两个PDG result/log、同source fixture `kv-pdg-medium.json`、上述CPU consumer、
诊断job的diagnosis.json和`.local/diagnose-weighted-pdg-rounding.py`作为最小复现；不修改输入/容差。

原profile初次因NPU初始化临时kernel_meta污染cwd而exit2；修复fea42aa把cwd提前放到输出目录。
profile2数值通过但有周期结束警告；ceae7bf显式完成采集周期，medium两项已通过。
原GDB诊断留下的9个CANN scratch文件已在确认旧进程终止后带hash移至原诊断job/cann-kernel-meta，未删除。
详情与原失败hash见对应报告，不恢复旧队列。

## 下一步

1. 提交/推送本次性能证据和导航修正；随后继续工作，不暂停。
2. 补小型runtime冷启动与standalone内存观测入口，使用新固定源码/新目录；不重复完整性能清单。
3. 实施一般单样本跨卡节点/状态放置：保持完整区域选择/Attention、物理边与原始时钟、私有state，
   设备请求/响应及共同commit，完整VJP/别名归约/optimizer发布与精确恢复分别验收。
   可复用`peer_exchange`、`remote_state`、`task_services`，新weighted路径尚未写代码。
   `weighted_flow_internal.h`、`weighted_flow_local.cpp`、`weighted_window_snapshot.cpp`、`weighted_window_vjp.cpp`
   是目前已定位的单设备边界；不得把样本副本包装成单样本切分。
4. 最终核对公共接口/模块能力/追溯与支持矩阵。既有严格数值见证和CUDA真机清单单列。

检查：`git status --short --branch`；`python scripts/status.py`；`python scripts/check_upstream.py`。
任务复核：`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；停止`systemctl --user stop UNIT`。

## 环境与保留限制

实际cwd `/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，可用磁盘约3.3TiB。
现行规范 `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`、`/etc/modulefiles`；旧guide失效已获用户授权采用现行规范。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK：module load libtorch-npu/2.10.0-cann9.0.0；Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH追加本项目.local/venv-core5/bin供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0、ATen/BLAS1；保留CANN PYTHONPATH。
长任务固定源码、独立background.slice、有限时长，不编辑活动checkout。STATUS用durable_records原子写并回读。

保留：eager FP16 GDN默认AdamW约.001984 strict xfail；默认异步双NPU Python VJP CANN invalid event
（launch blocking通过不解除）；SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6、mixed-a同见证。
CUDA所有真机范围待硬件，不以CPU或编译冒充。公共环境与只读参考仓库未改。
