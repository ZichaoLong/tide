# 当前交接 — 五记忆张量窗口与公共消费者迁移（2026-11-11）

授权按[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)继续。
分支graph-execution-foundation，固定数学0770593838a37c790a568226930567ed60f99b83；19快照核验通过。

## 已完成

- `01cde3b`固定张量整链资格和FC4/FC16三模式成本、独立profile：
  [证据](evidence/tensor-windows-20261111.md)。FC16 native训练CPU1/NPU为1.2863/0.6788秒，
  直接FC1.3706/0.6978秒。原始记录已含warm TTFT 0.44835/0.22037秒，报告解释已修正。
  小图资格不放大成完整FC16严格训练资格；跨设备Full/消息有差异。
- `9fa852c` KV相对位置/距离、int64大时钟、六个安装消费者训练/精确恢复通过，
  [证据](evidence/column-positions-20261111.md)。
- `b860e6a` native affine/delta/GDN/SSD、固定时间卷积、typed私有状态和ragged容量：
  [固定资格](evidence/recurrent-columns-20261111.md)。CPU279+27 passed，13原数值xfail；
  24个CPU/NPU安装消费者全部strict numerical/recovery passed，独立SDK五记忆CPU通过。
  `column-recurrent-qualify`及`column-recurrent-npu`已终态passed；无活跃任务。

## 接续动作

1. 固定证据独立提交并推送；原实现/失败身份保留。
2. P5/P6：检查公共实验内置任务的张量输入、loss、输出及实际分派，避免代表用例继续经过逐事件边界。
   对独立任务、完整事件观察、非Settle及放置后备保留明确能力身份。
3. 相关训练/精确恢复后，有限宽FC、176节点HB/LH、sum、KV/SSD层链及线程对照。
   mixed-a/b、resident、多设备新列迁移及约1B/9B仍按ROADMAP推进；GPU真机延后。

## 可用现场

环境`artifacts/column-env.sh`，私有`artifacts/additive-work/env`，CANN9.2/Torch2.12。
固定worktree `artifacts/columns-20261111/qualified-recurrent`；native/独立SDK构建
`artifacts/columns-20261111/recurrent-dev2/{build,build-sdk}`，复用hash见`recurrent-build-reuse.json`。
安装包`install-recurrent`，消费者`recurrent-{cpu,npu}-*/result.json`；作业在`artifacts/jobs/column-recurrent-*/`。
可用CPU/内存和空闲NPU可用；共享NPU须当前授权。长任务冻结源码并使用有界background.slice。
用户两个未跟踪TL-REQ-004/005交接文件原样保留、不提交。实际修改见git status。
