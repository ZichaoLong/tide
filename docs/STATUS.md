# 当前交接 — 区域张量资格与resident固定tile后继（2026-11-12）

固定数学0770593838a37c790a568226930567ed60f99b83，19快照通过。
分支graph-execution-foundation；用户授权整体重写、阶段提交/非强推及本对话共享NPU。
mixed-a/b/c是可简化放置预设，不是架构层；GPU延后。三份用户TL-REQ-004/005/006 handoff只读不提交。

## 已固定与复现

实现1f0e6e43已推送；[固定资格/38点成本/静态审阅](evidence/region-tensor-20261112.md)。
Python/C++区域均值显式块VJP、C++整组KV写入、跨节点Read保留共享传输；两套core/resident构建通过。
CPU304通过/9原数值xfail，六SDK CTests，12安装流程三AdamW/新进程raw精确恢复通过。
独立SDK NPU五记忆×三混合放置列检查通过。resident9局部与optimizer32轨迹/256更新通过；
完整消息组合仍录入EE1019。一次trace证实14投影/140容量body，失败前929vendor op，末task2043。
未运行依赖完整flow/owner/规模；原失败不改写。317安装Python文件与固定源码一致。
FC16默认native CPU1/NPU-host2完整训练1.1854/.2856秒；LH176为1.3900/2.8407秒，直接.7796/.6121秒。
IndexCopyBackward2066→8，含native私图23333→18140，但来源payload复制/阶段重装配仍有明显差距。
38文件静态审阅定位下一主机边界为source-owner/region-window typed上下文，见ROADMAP。

## 当前未提交后继与下一动作

正在删除resident bank容量树，使用设备bank+tile游标复用单一数值体；保留single-bank快路、
稳定跨tile dW、结构连接位与实际storage计费。后继文件由resident_rewrite支线修改，主代理统一审阅/提交。
新padding界须明确为P<A+G*T，不继续冒称旧小于2倍界。完整消息用例原样复验；
若该候选仍录入失败，停止相同候选，再设计有界program组合，不加预算或缩小原输入。
当前没有该后继的构建/设备/性能资格。P1—P6未全部完成，剩余范围只在ROADMAP。

下一命令：git diff --check；运行tests/test_resident_weighted_bank_gemm.py及相关开发检查；
提交稳定后继，在新的/dev/shm工作树重建两套resident（复用core须显式核对build manifest源码身份，
否则重建），按projection→messages→optimizer/session→安装训练/新进程恢复→原规模逐层推进。
现有qualification.py/resident-scale.py及原配置在artifacts/region-tensor-20261112/可作启动模板；
新作业必须新路径、固定新源码，不修改已固定输入。主机未变则不重复本轮38点计时。

## 作业与资源

1f0e6e43所有11个服务已终态；构建/CPU/安装/测量/profile及独立组件通过，resident-sdk和trace失败。
无活跃持久作业。状态/日志artifacts/jobs/region-tensor-*；记录和计划artifacts/region-tensor-20261112/。
固定源码和构建/dev/shm/gef-region-tensor-20261112/{source,core-native,core-sdk,resident-native,resident-sdk,install}。
不要混用Python-owned与SDK。运行环境ascend/dev-workspace-9.2.0和独立libtorch-npu/2.12.0-cann9.2.0。
构建初始AllowedCPUs未生效，已纠正并记录；后继用taskset真实继承CPU池并核对/proc，不能只看请求属性。
全部可用CPU/内存与合法分配NPU可用，避免项目重型构建与计时争用。主盘约825GiB、shm约280GiB余量；
用户319GiB/软300/硬800GiB且仍6日宽限，无磁盘阻塞。不得删除活跃输入或参考仓库。
重新进入：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
