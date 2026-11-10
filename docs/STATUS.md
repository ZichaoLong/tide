# 当前交接 — 联合prefill局部合批与部署回归（2026-11-10）

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
用户批准持续实施/推送、全部可用CPU及空闲/有余量共享NPU；数值有限诊断后分列推进。
参考/下游只读，无子代理。两份未跟踪TL-REQ-003/004用户handoff原样保留、不提交。
GPU按授权延后；下游依赖升级/学习验收尚未执行。唯一待办及停止条件见ROADMAP。

已交付分量窗口、联合KV准备、批量输入守卫、整序列VJP、query完整根合并与按需trace；
导航见prefill-regression.md、semantic-map.md和evidence/README.md。
34项CPU/NPU代表运行及精确下一窗恢复已记录于a841860；实际大尺度独立数值not_checked。
原FC16/B8/L128/d128的a553/fb完整返回CPU29.788/NPU55.482秒，最小返回22.068/31.822秒；
不同返回契约及设备实际Full/消息不混算等价加速，原任务学习/完整数值未认证。

f15c4a5按owner一次装配KV写伴随的固定任务gef-joint-prefill-owner-join已passed：
新Python-owned/独立SDK构建、CPU74/NPU37测试、两种安装消费者CPU/NPU三步训练与恢复，
SDK CPU/NPU完整/outputs三步及新进程恢复均通过；原始结果待纳入小型审阅证据。
日志artifacts/jobs/joint-prefill-owner-join/task.log；XML为owner-join-{cpu,npu}.xml。
新FC16最小返回成本及B2/L8 profile在artifacts/joint-prefill/{cost,profile}-owner-join-*。

当前源码改动：cat evidence共享GEMM；Read有限值按8MiB分组检查；分配入口数值守卫合批、
实际int64分支一次回读复用于前向/反向。新末行拒绝/不发布半成品/提交次数/FP64字面梯度测试。
开发Python45通过/2严格数值xfail，分配/正支持/结构/输入守卫127通过；C++尚待新构建。
逐行BMM候选同样触发指定约1.044e-6差异，未继续权重批复制。共享GEMM同材料FP64见证：
原标量对FP64最大9.06e-7，新批量1.38e-7；原严格失败保留、不提高容差。
下一步固定本实现，单独新建包和Python/native/SDK，核对相关独立根/训练/新进程恢复，
再补原FC16同返回成本/profile和少量代表；不能冒用旧构建认证新C++。

已有后继任务均使用独立冻结源码，不读取工作区：
- gef-joint-prefill-layer-anchor，固定930d1ba驱动/f15c4a5库/包，已failed。
  CPU13项与KV/SSD×三实现成本完成；NPU7项中3通过/4失败，后续NPU成本未运行。
  2项是测试跨设备比较元数据错误；2项SSD输出约3.07e-6超原阈值。须修复比较入口，
  有限完整根/FP64定位后独立补NPU成本，不将外层失败当作全部子项失败。
- gef-joint-prefill-deployment-resident，固定a553/core build-roots-a/resident-roots，已failed。
  FC16被tensor_bytes预检拒绝，SSD被单窗口retained预算拒绝，均尚未执行窗口。
  先按实际静态布局补显式预算再运行一次；不是设备数学或精确恢复失败。
- gef-joint-prefill-deployment-{cpu,single,dual}顺序运行/待前项，固定930驱动/f15库/包。
  CPU补wide-fc16线程/1.051B32线程；单NPU补FP16/异构LH/1.051B；双NPU补FC16/9.314B。
  大尺度推理B2/P8，其他B4/P16，每项一次预热/测量、最终窗口checkpoint和实际下一窗恢复。
  各RuntimeMaxSec36000，CPU配额320/32/64，内存512/384/640GiB；大尺度每卡须50000MiB空闲，
  准入最多等60秒。驱动身份见artifacts/joint-prefill/deployment-driver.json。

各任务日志artifacts/jobs/joint-prefill-NAME/task.log；结果artifacts/joint-prefill/deployment-PART/runs/result.json。
实际活动与终态以python scripts/status.py为准；不要把waiting/running记通过。
构建与正式计时不重叠；源码只读worktree、安装及库保持原身份。当前磁盘813GiB空闲、quota213/300/800GiB，
大写入/新9B checkpoint前仍须重查；不清理其他项目/公共工具链，也不停止其他项目作业。

起步：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
检查/停止：systemctl --user status/stop准确gef-joint-prefill-NAME.service，复查整个cgroup终止。
正式资格先固定实现，再提交证据。继续审阅每个inner结果、记录正负经验和准确复现入口。
