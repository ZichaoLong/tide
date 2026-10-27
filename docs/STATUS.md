# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
总体未完成，阶段提交后继续。[ROADMAP](ROADMAP.md)为完整待办，[execution-flows](execution-flows.md)为持续契约。

## 已实现与固定证据

一般SettleGraph、SrcClass/int64、五记忆/逻辑时间短卷积、正式配置、实际packed/VJP/计算共享、
CPU完整训练与恢复从[证据索引](evidence/README.md)进入；定向资格不认证未测矩阵。
470ce06/01689c0 Python/native有限mixed、显式同步双卡及master恢复；670ead3 standalone masters/schema2
CPU24、NPU FP16显式SGD24图/AdamW2图和CPU handoff通过。CUDA新栈core构建/CPU24通过，无GPU实机。
常驻Alloc/发送/head/Region/Aggregate/来源低秩/canonical参数/raw state/固定时间卷积已逐项固定资格；
均为局部组件，不代表完整weighted resident图。旧失败及精确范围见各项证据，不自动重跑。

最新实现2af401dc0b2912fcf09faac9ab1f31a43aec489a：共享投影/可选RMS与bias。
固定资格CPU4（1.03秒）、NPU四配置48行/三bank/完整VJP/None零/保存版本与容量；CUDA sm80/90/100构建通过。
证据[evidence/core5-resident-projection-20261027.json](evidence/core5-resident-projection-20261027.json)。
core5-resident-projection-{npu,cuda}-qualify均clean/exit0/inactive；无活动reader。
NPU保留开发treee741c50847de858517264e9546eb503e20d36cb4原manifest并核验全部摘要。

## 当前工作与下一动作

681a7e9 KV Upd固定资格通过：CPU24（2.63秒）、NPU72行/三bank/五字段完整VJP及CUDA三架构构建。
core5-resident-kv-update-{npu,cuda}-qualify均clean/exit0/inactive；证据evidence/core5-resident-kv-update-20261027.json。
开发tree28f8ee2ef2fe9201eea8f80a442849c352f074cf，原manifest保留且摘要匹配。

afe7b5e pool固定资格通过：CPU14（1.20秒）、NPU四配置144行完整VJP/独立返回根/保存版本及CUDA三架构构建。
core5-resident-pool-{npu,cuda}-qualify均clean/exit0/inactive；证据evidence/core5-resident-pool-20261027.json。
开发tree202f8b0f683f115d93d8c74e6bb27df8d75f82d5；NPU保留manifest并核验全部摘要。
原dev/dev2/dev3失败、FP32 clamp边界诊断及CPU adapter reset问题均按原树/失败状态保留在证据中。

fea6fa7 KV window/sum固定资格通过：CPU20（1.95秒）、NPU两窗口120行/四Mem根完整raw VJP、
原始int64快照/保存版本及CUDA三架构构建。core5-resident-kv-sum-{npu,cuda}-qualify均clean/exit0/inactive。
证据evidence/core5-resident-kv-sum-20261027.json，开发treecac7cbde71943a29db7f09487c7d013665e74a3e。
CPU adapter原exit139及invalid horizon见原诊断记录，未改历史终态。

rotary开发tree0027114a70778f701e019a1e00b951a51e5e922e：CPU4通过、NPU构建通过但极大age旋转错误，
tide-core5-resident-rotary-dev-20261027.service failed/exit1，原source/log/status保留。
观察符合int64数值转换截成低32位，正分开检查相位和厂商Sin/Cos；不改测试输入/容差。
已实现显式IEEE最近偶数舍入的非负int64→FP32 helper，同时用于窗口衰减与旋转；
CPU新增转换舍入边界、窗口+旋转共11项通过。NPU窗口新增真正极大age（此前只有极大绝对时间与小age）。

rotary-dev2冻结tree952c7789fb90f88ab3310cc73bc4773fd0e35f0d：CPU11、NPU构建通过；
实际FP32 phase与CPU逐位相同，但厂商Sin/Cos在巨大相位下结果错误，旋转不保范数。
unit tide-core5-resident-rotary-dev2-20261027.service failed/exit1；后续kv-sum未执行。
已补有限FP32全指数范围的Payne–Hanek周期归约，再调用厂商Sin/Cos和象限还原；
CPU独立math.sin/cos与窗口合计13项通过，设备尚未验。原失败/source/log/status保留。

rotary-dev3冻结tree7207e6dd66b334935329d3ce896863b1f4f38702通过：CPU13、NPU窗口/sum与旋转完整检查。
tide-core5-resident-rotary-dev3-20261027.service inactive/success/exit0；原source/log/status保留。
本次提交旋转、周期归约和大age数值转换修复；此前两个开发失败仍保留failed。

下一动作对本实现提交运行tide-core5-resident-rotary-{npu,cuda}-qualify-20261027.service，结果未知。
使用冻结artifacts/worktrees/core5-resident；NPU复用dev3产物并核验全部摘要，不改原manifest。
输出artifacts/jobs/core5-resident-rotary-{npu,cuda}-qualify/{status.json,task.log,checks/result.json或control-build.json}。
launcher .local/core5-resident-rotary-{npu,cuda}-qualify{,-workload}.sh；两者检查weighted-kv-sum和weighted-rotary。
background.slice/Nice10、NPU600秒/CUDA1200秒、2build workers/BLAS单线程、NPU queue180秒。
检查systemctl --user show上述unit及durable记录；停止systemctl --user stop上述unit。失败不标passed。
通过后证据另提交推送，继续qry、Describe/Full和完整图。

主工作区另有weighted_kv_query{.h,.cpp,_vjp.cpp,_internal.h}、query plan/roots kernel草稿；
主工作区已补CMake/registry、契约和CPU依赖测试7项通过，尚未编译，不进入rotary提交/资格树。数值链为window→Q/K/V→read RoPE→两通道Attention→四返回；
VJP按独立返回根回传各参数角色/私有u/raw。后续继续qry、Describe/Full、四循环及完整图生命周期。
KV sum证据c198a43已推送，不修改活动reader的冻结源码。

完整weighted resident仍缺五记忆数值/Mem/Describe/Full、图在线就绪与推进、retained训练/optimizer、
精确checkpoint/跨卡；继续局部组件与完整生命周期，不借旧ContentFlow或CPU捕获轨迹宣称完成。
随后有限profiling/性能选型与最终追溯，不做全开关笛卡尔积。

## 未解除的严格限制

- FP16 GDN默认AdamW：Python/native nodes.5.kernel.conv.weight严格更新差约.001984，专用strict xfail；
  只接受该字段，未改容差或输入，SGD通过不解除。原175pass/2fail与95pass/2fail任务保留failed。
- 默认异步双NPU完整Python VJP：CANN invalid event；显式ASCEND_LAUNCH_BLOCKING=1通过不解除。
  core5-peer-event及current0/1/serial复现保留，不恢复旧队列。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a相同；原失败/诊断保留。
- CUDA实际GPU单卡/跨卡、完整训练/恢复、常驻与性能待硬件，CPU或构建不冒充实机。

## 环境与操作

实际路径/mi/data2T/zlong/home/llm/graph-execution-foundation，zlong，约3.3TiB空闲。
现行Ascend规范/opt/software/ascend-locks/dev-workspace-9.1.0/README.md及/etc/modulefiles。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate（Python3.12.13/Torch2.12/CANN9.1）。
SDK独立shell：module load libtorch-npu/2.10.0-cann9.0.0；sdk_python=$LIBTORCH_ROOT/../../../../bin/python，
PATH前置其bin，ACL_OP_INIT_MODE=0，PYTHONPATH追加项目python但保留module提供的CANN路径。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh，Torch2.10+cu128/Python3.11/Toolkit12.8.1；
服务PATH追加项目私有Ninja但保持CUDA Python优先。公共包与工具链未修改。
长任务background.slice/Nice10、2build workers/BLAS单线程、新输出/有限时长；
先更新STATUS、固定源码，不编辑活动reader。源码/构建/实际设备/性能分别记录。
检查systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus及status.json/task.log/checks/result.json；
停止systemctl --user stop UNIT。STATUS用scripts/durable_records.py原子更新并回读。
