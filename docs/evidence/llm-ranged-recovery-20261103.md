# 分段下载恢复（2026-11-03）

固定实现`8b45cb42325224e38f945405988f5efe0bd83e3e`，公共数学未变。
仓库外固定worktree的asset/matrix/queue相关检查27 passed。
实际HTTP两段各1MiB，206与起止/总长正确，约4.0/4.3秒；不外推完整下载时间。

实现按32MiB请求精确Range，校验总长/编码，最终仍核对官方hash。
每文件最多6次临时错误重试、有限退避，成功分段不重置失败预算。
旧asset-state按内容hash保留；错误Range、hash、配额等不重试。

首组9a69180服务因attempts/recovery目录未创建而退出，无新增下载和遗留后代。
8b45cb4修复目录准备与保全回归后以新身份启动，原失败保留。
`gef-lm-assets-small-20261103-d`与`gef-lm-assets-moe-20261103-d`在background.slice运行，
各30小时、最多两流，project160GiB/account300GiB停止线。
本报告观测两服务running，不能宣称资产完成或MoE已测。
[JSON](llm-ranged-recovery-20261103.json)保留快照，当前状态见[STATUS](../STATUS.md)。
