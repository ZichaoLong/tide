# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83；授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
已推送87e4031。总体未完成，下一阶段见[ROADMAP](ROADMAP.md)，不在阶段提交后暂停。

## 当前实现与证据

一般SettleGraph、结构化SrcClass/int64、五记忆/短卷积、正式发送与评分、
实际packed/VJP/精确计算共享、standalone完整训练/checkpoint的分阶段CPU证据
从[证据索引](evidence/README.md)进入；各定向资格不等于全部矩阵。

[470ce06 FP32设备](evidence/core5-device-20261027.md)：CPU200，Python/native mixed-a/b各2、
mixed-c各19、显式同步两卡各2组合。默认异步及SDK严格见证未解除。
[8c8824f half CPU](evidence/core5-half-20261027.md)：99passed/2strict xfailed，后两项不计通过。
归一化/Attention内部FP32、一阶VJP、int64先相减的长时差与KV位置、独立公式及训练恢复。
[01689c0 half设备](evidence/core5-half-device-20261027.md)：CPU12；Python/native各mixed-c
AdamW4+SGD2、mixed-a/b各1、显式同步两卡各1，均完整反向/四步/raw与master精确恢复/CPU handoff。
native half profile35945kernels/1222hostscalar/1368memcpy，32candidate/31Full/44messages，
20设备AiCPU；是有限算子归属调查，不是吞吐排名或完整FP16配置认证。

## 当前 standalone master 增量

准备提交实现：OptimizerPrecision显式FP32 masters/静态loss scale，None/零/alias保持；
原参数/完整图格式schema1读取兼容、schema2新增Half原值和master/scale。Graph v13、Python checkpoint v5不变。
完整preflight、精确raw/slots恢复、独立Torch更新、有效checksum但不一致master拒绝及standalone设备→CPU续接入口。
所有源码/契约见git status；这些内容不在01689c0的设备资格内。

开发门禁均已终止：
- native-master-dev：tree0f337e7cfa7d4fa79b0d468b5519d799cf6a8719，20passed/exit0；
  含FP64/FP32各24图、half显式SGD24图及AdamW2图，四步和完整raw恢复。
- native-master-final-dev：tree560b91b6a41dd190afc2790031dd18cf3776e25d，9passed/exit0；新增拒绝/精度报告。
- native-master-handoff-dev：tree18e7e04d7ee3fd64ff9862f1b21b2927bed601d1，9passed/exit0；编译最终设备→CPU验收分支。
units为tide-core5-上述名-20261027，output artifacts/jobs/core5-上述名，manifest已保存。
bootstrap当前为最后开发快照，device干净01689c0，sdk干净470ce06；当前无reader。

下一动作：提交这项实现后冻结精确SHA到bootstrap/sdk；CPU固定资格运行
`tests/test_cpp_master_optimizer.py tests/test_cpp_optimizer.py tests/test_cpp_checkpoint.py tests/test_cpp_weighted_training.py`，
最终完整24项。SDK独立栈重建并检查loader，再执行half/FP32 scoring/显式SGD及AdamW有限完整图资格。
先实现提交、后固定资格、再单独证据；不推定SDK或新native格式已获设备通过。
之后继续weighted resident、CUDA源码/构建/真机分账、有限性能及最终追溯。

## 未解除的真实失败

- half GDN默认AdamW：Python/native nodes.5.kernel.conv.weight更新严格差异约.001984；
  专用GDNAdamWParityError strict xfail，只接受该字段；显式replay/标量舍入顺序诊断仍失败。
  未改输入/阈值，SGD通过不解除。half-normalization-dev175pass/2fail、half-clocks-dev95pass/2fail保持failed。
- 默认异步两卡完整Python VJP：CANN invalid event；core5-peer-event及current0/1/serial复现保留。
  显式ASCEND_LAUNCH_BLOCKING=1的FP32/half通过不解除默认异步限制。
- standalone SDK2.10原FP32门禁：fc/affine第三轮evidence误差3.09944e-6；mixed-a相同。
  前两轮forward/grad/update通过，小梯度经AdamW累积；未执行图不推定通过。
  core5-sdk-affine/core5-sdk-affine-drift诊断及原failed状态保留。

## 环境、资源与操作

实际路径/mi/data2T/zlong/home/llm/graph-execution-foundation；zlong，约3.3TiB空闲。
现行规范/opt/software/ascend-locks/dev-workspace-9.1.0/README.md与/etc/modulefiles；旧指南已不存在。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate。Python3.12.13/Torch2.12.0+cpu/Torch-NPU2.12.0.post2/CANN9.1/GCC10.3.1。
SDK单独shell：module load libtorch-npu/2.10.0-cann9.0.0；ACL_OP_INIT_MODE=0；
PATH前置$LIBTORCH_ROOT/../../../../bin提供匹配NumPy，显式sdk_python使用同栈Torch。
公共包未修改。长任务background.slice/Nice10、ATen/BLAS单线程、2build workers、新输出/有限时长。
活动reader源码不可编辑；复用前核验C++及全部binary摘要，保留原manifest的build source字段。
检查python scripts/status.py；systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus；
读取status.json/development.json/task.log。停止systemctl --user stop UNIT。
不恢复旧队列，不重跑旧矩阵；失败保留，源码/构建/实机与性能证据分别记账。
