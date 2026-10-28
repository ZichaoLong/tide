# 当前交接 — 2026-10-28

授权分支 `graph-execution-foundation`；分配/诊断 `56631eb9c48fc006d0df00b3a259aee10e04314b` 已提交推送。
固定数学 `0770593838a37c790a568226930567ed60f99b83`；19份core-5/5份core-3快照核验通过。
当前低秩KV增量即将固定提交。用户授权实现、有限实验、分阶段提交和普通推送，提交后继续。
参考仓库/公共环境只读，不用子代理，不恢复旧队列。STATUS为唯一交接，ROADMAP为唯一待办。

## 当前实现与开发验证

quad按cap质量分位评分选局部原点，并在约束分支消去共同阈值；Python有限排序、C++独立二分、
设备32次整数key查询保持不同求解程序。实际失败输入及120位Decimal分析已经捕获，
两个最小失败行存于tests/data/quad-{wide2048,matched2048}-failure.json。原守卫、dtype、精确解VJP均保持。

所有eager KV行统一为base@x+U@(V@x)，省去2d×d稠密adapter临时矩阵；未共享packed复用归一化。
resident稠密矩阵/tape保持独立验收范围。手算Jacobian、多维独立dense公式/全部字段根为数学锚点。
本机aarch64/Torch2.12 CPU算子探针通过；静态可移植性扫描1692文件，新增四个实现文件无匹配项。

`core5-factorized-kv-dev-20261028` / `artifacts/jobs/core5-factorized-kv-dev` 正在运行，不能记为通过。
源码为56631eb加冻结低秩修改；工作树`artifacts/worktrees/core5-capture-dev`，精确快照见job/source.tar.gz。
开发定向198 passed/2 strict xfail；本次新补多维公式测试在主树20 passed。
任务随后构建standalone benchmark、跑二进制材料测试、原wide2048及Python/native D64两步诊断。
unit在background.slice，Nice10，ATen/BLAS线程1、build2、MemoryMax128G、总50分钟。
日志`artifacts/jobs/core5-factorized-kv-dev/task.log`，终态`status.json`，诊断`study.json`。

上一`core5-local-allocation-dev`终态failed/exit1：构建/定向432项、材料5项通过，wide完整测量窗
执行1.39249s但keys最大差5.34058e-5严格失败；D64离散结构一致，完整返回/梯度/SGD仍不等价。
这是旧源码的原始结果，后续不覆盖。局部四次实际KV操作数dense/factored最大差6.67572e-5，
FP64误差诊断分别约6.16676e-5/3.92842e-5，只说明局部算术，不是大图或吞吐资格。

## 下一动作

1. `python scripts/status.py`；`systemctl --user show core5-factorized-kv-dev-20261028 -p ActiveState -p SubState -p ExecMainStatus`；检查上述日志和每个子结果。
2. 固定本次实现提交，以干净源码执行CPU相关资格；NPU allocation/flow/config/training/node-training及eager KV，CUDA同源构建/CPU核验分别记录。
3. 所有项目构建/资格终态后，原wide和core-3主体参数匹配材料作一次有限CPU对照；D64原严格门禁继续保留，完整通过才称等价加速。
4. 更新经验、对应契约、固定证据、STATUS/ROADMAP，提交并普通推送；无硬件范围保持独立待验。

## 已有证据与限制

[07e56ca固定资格](evidence/core5-private-projection-20261028.md)：CPU574 passed/2 strict xfail、NPU相关单/双卡训练恢复、CUDA构建/CPU核验。
[性能设施](evidence/core5-performance-transfer-20261028.md)、[数值/异步隔离](evidence/core5-numerical-isolation-20261028.md)、[经验](performance-lessons.md)。
原matched2048约92.62亿主体参数，旧测量残差拒绝；新完整材料尚未复验。
D64 FP32完整记录/梯度/SGD、eager FP16 GDN AdamW、SDK FP32 affine、默认异步双NPU、FP32反馈PDG严格失败分别保留。
CUDA真机、x86_64、大型设备完整训练/跨卡吞吐待验。当前资格不能外推全部拓扑、配置、规模或平台。
