# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
总体未完成，阶段提交后继续。[ROADMAP](ROADMAP.md)为完整待办，[execution-flows](execution-flows.md)为持续契约。

## 已实现与固定证据

一般SettleGraph、SrcClass/int64、五记忆/逻辑时间短卷积、正式配置、packed/VJP/计算共享、
CPU完整训练与恢复从[证据索引](evidence/README.md)进入；定向资格不认证未测矩阵。
Python/native有限mixed、显式同步双NPU、half/master及standalone恢复范围见ROADMAP；严格失败保持。
常驻Alloc/发送/head/Region/Aggregate/来源低秩/canonical参数/raw state/卷积/共享投影/实际KV Upd、
pool、sum/qry Mem及KV Full已逐项固定资格；均为局部组件，不代表完整weighted resident图。
CUDA源码/三架构构建通过范围见各报告，无GPU真机；不自动重跑历史矩阵。

本段最新固定资格：
- 1eb4b72：旋转/周期归约与真实巨大age数值转换。CPU13、NPU rotary及KV sum完整回归、CUDA三架构。
  evidence/core5-resident-rotary-20261027.json；开发dev/dev2设备失败均保留，dev3 tree7207e6dd66b334935329d3ce896863b1f4f38702通过。
- 8b1be80：完整qry Mem。CPU38、NPU120行四根及全部参数/输入/rawVJP、CUDA三架构。
  evidence/core5-resident-kv-query-20261027.json；dev启动脚本路径错误exit4保留，dev2 treeef1c888bdb4a9d39f085deaeda7e7e25fb59f062通过。
- 386970a9dcd0b826e390be020c2a86ea0c7a3437：公共Full残差与多头KV Full。CPU50、NPU residual48行/Full96行，
  全参数/input/raw VJP、保存版本、None/零及容量；CUDA三架构构建/loader。
  evidence/core5-resident-kv-full-20261027.json；开发tree5c4d2c92c6335c6c330d7254376e5187b017dbaf通过。
  core5-resident-kv-full-{npu,cuda}-qualify均inactive/success/exit0；无活动reader。
所有NPU固定资格保留原开发manifest并核验全部源码/core/binary摘要。

## 当前工作与下一动作

cd576d24434c5daaa21f57d316a93b9bf1c91f0a完整Describe固定资格通过：CPU38/3.86秒，NPU八配置，CUDA三架构构建。
证据evidence/core5-resident-describe-20261027.{md,json}；两个资格unit均inactive/success/exit0，无活动reader。
开发dev2 tree88c5895efb483c4b0470a60b76b893c6d28e8d00，动态缺席→出现回归通过；原dev记录保留原范围。
当前主工作区正在实现四循环数值：row algebra矩阵/除法/exp/sqrt/softplus/卷积、
原始时间快照与affine/delta/GDN/SSD的Upd/Mem/Full；这些修改未提交、未构建，尚无通过结论。
下一动作完成独立直接公式与字段根/参数VJP测试，冻结开发树进行相关CPU与NPU检查，随后固定提交资格。

接续四循环的实际数值Upd/Mem/Full、完整在线图就绪/选择/推进、retained训练/optimizer、
精确checkpoint/跨卡；不能借旧ContentFlow或CPU捕获轨迹宣称完成。
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
