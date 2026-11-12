# 当前交接 — LH张量贯穿、REQ-006与resident块（2026-11-12）

本轮已实施、验证并整理固定证据；没有活跃作业。用户已授权继续ROADMAP P1—P6，
允许整体重写、常规构建/测试及分阶段提交推送当前分支。P1—P6总体仍未全部完成。
数学固定0770593838a37c790a568226930567ed60f99b83，19份教材快照检查通过。
实现基线7d5e05c3；历史资格保持各自源码身份。三个用户TL-REQ-004/005/006 handoff未修改、不纳入提交。
下游、ObsidianVault和参考仓库保持只读。

## 本轮交付与证据

- [节点/区域合批及成本](evidence/node-region-batching-20261112.md)：78fc90b4相容KV准备、96cb570a实际候选数区域分组、
  a95b8987有界unit transport检查。96cb CPU178通过/9原数值xfail；a95 CPU150通过；
  CPU/NPU Python/native × KV/SSD共8安装消费者训练/新进程精确恢复通过。
  原NPU定向59通过/12设备绑定失败，e3ac44fb显式测试placement后3f99有15项通过；原失败保留。
- 同一LH176/d64/B4/L16、61有效输入位置，默认native训练CPU1为2.451秒，NPU为5.011秒；
  直接公式0.849/0.661秒。区域调用307→94，候选/Full/消息仍8368/2572/15340。
  本轮CPU/NPU共26成本点；NPU为共享条件。FC16/B8/L128默认训练CPU1/NPU为1.278/0.662秒。
  [f681/3f99尺度检查](lh-tensor-baseline.md)：CPU FP64数值投影通过，CPU/NPU FP32有限严格失败；
  离散支持/int64通过，36个None/连接零差异另列。没有认证该尺度全部中间根、多步optimizer/恢复。
- [REQ006四代表诊断](evidence/req006-diagnostics-20261112.md)：固定下游6e7bd49e，四measure完成；
  原profile解析失败保留，2927ba40离线恢复。HB88全部704个AICPU核追到clamp_min Scalar反向。
- [常量重载与初态配对](evidence/req006-floor-20261112.md)：原Scalar相等边界与dtype阈值公式不符，
  7d5保留strict失败仍观测约5.972ms和2个AICPU；显式FP32常量Scalar/Tensor约0.263/0.298ms、各4个AIVEC。
  常量表示/边界VJP不同，不作无条件生产替换或整模型收益。f681的40检查和三进程初态/最终更新hash配对通过；
  --initial-state-out/in已公开，固定小fixture可独立复现。
- [resident块资格](evidence/resident-blocks-20261112.md)：a95独立调度/Read-selection与7d5的KV Full/messages
  共三个standalone NPU FP32组件通过，CPU元数据17项是前述150项的子集。
  新消息组合保留独立Read/proposal、7组根、真实alias、(c,G)/显式cG、saved reverse和10项拒绝。
  仍不是完整图owner/训练/续接；旧FC16 EE1019/507002和普通异步跨NPU EE1017未解除。
- [审阅清单](evidence/rewrite-followup-reviews-20261112.json)绑定14文件hash；5个Python文件43定义，
  9个C++文件为具名人工审阅，未做本轮C++自动函数提取；不宣称全仓已审。

## 下一步

1. 按[ROADMAP P5](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)实现跨节点Read/Full边界：
   从column_modules/read_placed/full_columns进入，按真实模块/私有bank/设备与预算成组；
   位置旋转避免查询维展开，先验证原始int64时间、独立未用根/连接零及非空左态。
   Aggregate的相同来源/真实参数alias复用另处理，不用普通stack掩盖None根。
2. resident从已验组件继续实际Aggregate/Upd、逐边/端口路由、device choose/commit数值循环与共享KV bank；
   然后独立中间根、跨窗tape、optimizer/checkpoint，先小图再原FC16预算。不得继承旧owner资格。
3. REQ006候选属于下游显式浮点floor边界；任何替换先固定常量/VJP契约及完整代表配对，当前没有下游验收。
   FP32尺度超差保留，只有新的首差/算术或实际消费者证据时再诊断，不原样重复矩阵。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
rg -n 'read_placed|full_columns|linear_blocks|rotary' cpp/src/weighted_column_query.cpp cpp/src/weighted_settle_columns.cpp python/tidegraph/weighted
```

## 作业与资源

所有本轮unit已终态、MainPID=0且ControlGroup无后代；生命周期在artifacts/jobs/rewrite-*/，
原始结果在artifacts/rewrite-20261112/，复现入口均已进入上述tracked报告/契约，不依赖旧raw目录。
最近任务：shared-cost passed；shared-qual failed（独立内项另列）；diagnostic-f681-cpu与diagnostic-3f99-npu
因LH FP32严格失败保持failed；floor-7d5因原Scalar严格失败保持failed；messages-7d5 passed。
原block-build缺-fPIC链接失败保留，a4b71bfc修复后新构建通过；diagnostic-cpu曾误用已存在c83目录而失败，
没有有效新资格，后继使用新路径并断言固定提交，不改写原记录。

固定源/build主要位于/dev/shm/gef-rewrite-20261112/{batched-*,diagnostic-f681-source,diagnostic-3f99-source,messages-7d5-*}。
复用的native/SDK核心均a95、cpp hash f4596708；新消息component为7d5、hash84a609f6。
大构建使用有界shm，持久记录留磁盘；本轮未因磁盘阻塞，没有清理参考仓库或用户数据。

用户2026-11-12已明确允许当前对话后续共享NPU，包括正确性、profile与性能；无需逐作业再询问，
不延伸为新会话默认许可。已将共享条件的有效测量、准入和争用记录写入优化skill与执行契约。
卡1/9的实际PID/start/uid、两次采样、映射和余量在policy/audit中。共享不代表独占峰值。
部分旧AllowedCPUs请求未形成结果中所写的进程亲和性，报告按actual记录；后继显式CPUAffinity/taskset，
仍须实际核对。构建可用全部CPU，正式计时避开自身重型构建，外部争用单列。

STATUS是唯一当前交接；未完成范围、停止与重启条件只在ROADMAP；提交身份与审阅结果见Git及证据索引。
