# 当前交接 — 本轮收尾，等待重新对齐（2026-11-12）

数学固定 `0770593838a37c790a568226930567ed60f99b83`，19份快照通过。
授权分支 `graph-execution-foundation`。**用户要求本轮之后停下汇报，重新对齐并获授权后再继续。**
本轮两条边界及必要验证已结束；不要自动启动下一轮优化、规模扫描或剩余P1—P6。
当前总体待办在[ROADMAP](ROADMAP.md)，本页是唯一当前交接。

## 已确认原则，不重新立项

固定core-5全局/局部数学、公共基座和代表性能是锚点。旧源码和接口可整体重写；
mixed-a/b/c只是可迁移的对照预设，不强迫架构分层。重入时遵守，不重复调查或执行同一原则。
保留一般图、局部替换、完整观察/独立根、训练和精确恢复。子代理/工具并行按收益选择，
主代理统一集成和资源调度。全CPU/可用内存可使用；本对话共享NPU已有授权，不能继承为新会话默认。GPU延后。

## 本轮交付与证据

- **c6afc20c主机Read/Describe→区域字段owner**：真实连续存储借用和块伴随已交付，
  [固定资格与20点成本](evidence/read-region-owner-20261112.md)。两套core构建、SDK7CPU/15NPU组合、
  12安装三AdamW/新进程恢复通过；CPU245 passed / 7原xfail / 2原FP32严格失败。
  原失败坐标及差8.8512897e-6保持，阈值未改。LH176 B4/L16/d64默认native完整训练CPU1/NPU-host2
  为1.419332/2.471804秒，直接.749660/.621882秒；FC16 B8/L128/d128默认native为1.156975/.301235秒。
  这是主机调度数值在NPU的结果，不是resident。主要LH差距仍在，不追加同一局部的无限优化。
- **0c974ccd resident唯一KV bank→Mem/Full→块伴随**：默认局部链已贯穿，
  [固定资格](evidence/resident-kv-owner-20261112.md)记录56 CPU、两套独立构建、11 SDK及4安装native设备进程全部通过。
  Read0/−/+、独立raw根、小图三步AdamW、两窗BPTT、实际raw解码和新进程精确恢复分别核对。
  `kv-bank-views-v1`：old/proposal默认有shape/present/valid，显式observe从真实bank解码，
  cotangents不要求先物化；完整Python记录和安装消费者已迁移。block checkpoint仍schema2。
  **新resident布局没有新吞吐或规模测量**；半精度、其他记忆、多owner、CUDA构建/真机未扩。
- 来源共享/规划与resident资源此前记录仍有效：
  [72e1d22c](evidence/source-bank-20261112.md)、[9ae62d0d](evidence/source-planning-20261112.md)、
  [08a349ce](evidence/resident-resources-20261112.md)。不重新执行已终态矩阵来恢复上下文。

## 尚未完成，等待下一轮对齐

P1—P6总体未完成。主机prepare/raw-bank→Read/Full和typed continuation仍有相邻交接工作；
LH直接/公共/默认的主要成本差距尚未闭合。新resident保留查询×条目表和部分标量位置/元数据核，
大尺度、有限window expiry完整设备图、更多模块/多卡及性能须独立验收；CPU局部expiry不能代替这些资格。
全仓逐文件/逐函数审阅未完成。详细限制、停止/重启条件只在ROADMAP维护。
本轮经验已归入[性能经验](performance-lessons-prefill.md#共享历史与显式观察边界)，
源码/数学/测试/实际命令从[bank契约](resident-kv-bank.md)和[复现导航](reproduction.md)进入。

## 作业、源码和工作区

无在途长任务；所有`resident-kv-owner-*`作业均passed/exit0，unit inactive、MainPID0。
两构建CPU0–159/160–319、各512GiB/2400秒；NPU资格物理11(native)/5(SDK)，idle准入，逻辑npu:0。
固定resident源码`0c974ccd4921999e6b7f6292888bfcb3478a33e1`在
`/dev/shm/gef-resident-kv-owner-20261112/source`，对应resident-native/resident-sdk/install/wheels同根；
复用c6 core前逐一核对cpp源码及二进制SHA。两套公开工具链独立，无公共环境修改。
记录`artifacts/jobs/resident-kv-owner-*`与`artifacts/resident-kv-owner-20261112`；
必要命令、结果、SHA已复制到仓库证据，不依赖旧运行目录才能复现。

用户的三份未跟踪`docs/tide-learning-lab-TL-REQ-004/005/006-handoff.md`仍只读、不提交。
除此之外无待集成的源码草稿。开发中的CPU harness签名/arena和W=1分支见证修正已保留于证据，
不能把开发/语法检查混成固定设备结果。磁盘未阻塞：最近主盘余807GiB、tmpfs276GiB，
账户337GiB/软300/硬800GiB、6日宽限；下一大写入前重查。

## 下一动作

向用户汇报并等待对齐/授权，不自动运行任何待办。重入只核验：
`git status --short --branch`；`python scripts/status.py`；`python scripts/check_upstream.py`。
历史提交按`python scripts/resolve_history_commit.py OLD_SHA`定位，见[迁移说明](history-migration.md)。
