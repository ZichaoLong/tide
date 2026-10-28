# 当前交接 — 2026-10-28

授权分支 `graph-execution-foundation`，实现基线 `07e56ca`，文档基线 `a7d6ecf`（已推送）。
固定数学 `0770593838a37c790a568226930567ed60f99b83`；19/5快照启动核验通过。
用户批准继续：捕获大型分配失败→独立高精度定位/修复→D64及训练差异→有依据的求解/批量优化→原材料/参数匹配测量→固定资格/提交推送。
参考仓库和公共环境只读，不使用子代理，不恢复旧队列，不修改旧失败终态。

## 当前增量

`core5-capture-dev-20261028` 已于10:54:21 UTC终态passed/exit0：156定向测试、standalone构建、两个原材料的失败快照取得。
这仅是诊断收集成功；`artifacts/jobs/core5-capture-dev/{wide2048,matched2048}/result.json`均仍failed/exit2，metrics空。
实际快照和独立120位Decimal分析位于同目录 `allocation-failure.json` / `decimal.json`。
wide2048误差3.3021e-5；matched2048在time26有20个候选，原残差1.7166e-4超过1.5259e-4。
两个失败行已原样复制为 `tests/data/quad-{wide2048,matched2048}-failure.json`，正式成果不依赖旧运行目录。

工作区实现了按cap质量分位评分选原点的quad求解；真实根落在[-tau*max(mu),0]，Python只排序可变带，
C++独立二分，设备用32次整数key查询选原点；保持原dtype/容差、原始响应和精确解VJP。
同时约束分支在原评分坐标消去共同阈值，避免先舍入scores-eta丢失评分差；松弛分支不变。
新失败快照、Decimal/成功调用观察、projection诊断及benchmark核对阶段峰值改进仍未提交。
纯Python定向138项及大共同shift22项通过；开发native/Python定向432项、二进制材料训练5项通过。
主树新增常驻源码CPU适配37项通过；NPU新求解真机/CUDA新构建尚未验证。

`core5-local-allocation-dev-20261028` / `artifacts/jobs/core5-local-allocation-dev` 已终态failed/exit1。
构建/上述定向测试通过；wide2048完整窗口已执行(单次1.39249s)但keys差5.34058e-5严格失败；
Python/native D64两步离散结构一致，完整返回/梯度/SGD仍strict_mismatch，不作等价加速。
复用已终态的开发工作树路径 `artifacts/worktrees/core5-capture-dev` 作下一冻结快照和增量build；
先前快照精确保留在旧job/source.tar.gz和tree hash，旧standalone安装及失败记录不改写。
上述原用例复验已终态。下一步固定本次数值/诊断增量；继续KV投影算术与性能增量，随后必要固定资格和大型matched2048测量。
资源：线程1/build2、Nice10、128G配置上限、50分钟；新status.json/task.log记录全部终态。

## 已有证据与保留边界

[最新固定资格](evidence/core5-private-projection-20261028.md)：CPU574 passed/2 strict xfail、NPU相关单/双卡训练恢复、CUDA构建/CPU核验。
[性能设施](evidence/core5-performance-transfer-20261028.md)、[数值/异步隔离](evidence/core5-numerical-isolation-20261028.md)、[经验](performance-lessons.md)。
原D2048/参数匹配仍残差拒绝；D64 FP32完整记录/梯度/SGD失败、FP64诊断通过不解除它。
eager FP16 GDN AdamW、SDK FP32 affine、默认异步双NPU和FP32反馈PDG严格失败保留。
CUDA真机/x86_64和大规模设备/跨卡吞吐待验；本轮尚无新通过或性能结论。
378旧job全部终态，最新private-study-cpu/npu为原failed；不要重标或复跑以恢复上下文。

## 接续

`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
新unit启动后用 `systemctl --user show core5-capture-dev-20261028 -p ActiveState -p SubState -p ExecMainStatus` 和job记录核验。
实际failure JSON与Decimal已完成；继续核对未共享KV稠密组合矩阵和factorized投影的完整差异，不放宽容差。
ROADMAP为唯一待办，execution-flows为持续契约。本轮改动尚未提交，正式资格先固定实现提交。
