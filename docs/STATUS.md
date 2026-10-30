# 当前交接 — 2026-10-30

新基座已发布并切换回本目录，分支继续叫`graph-execution-foundation`。
旧历史在远端`graph-execution-foundation-old`冻结于`83d384a4a5efb9df38ce77c395c2d9c5222b4e0e`。
数学仍固定`0770593838a37c790a568226930567ed60f99b83`，公共数学/API/容差与旧数值/异步/硬件限制不变。

[迁移资格](evidence/history-migration-20261030.md)：固定b0969fc，716提交逐一核对，181改号；
新CPU构建、13 CTest、两dtype smoke、177定向检查、4消费者单元/8安装恢复/4拒绝通过。
[发布证据](evidence/history-publication-20261030.md)：精确lease原子保存旧分支并替换新分支，独立远端核验；
新单分支clone约39.12MiB，大归档blob实际缺失，原文件/refs并发修改保护及目录切换通过。

没有活动构建/测试/发布服务。剩余清理：本提交保存必要证据后，删除同盘graph-execution-foundation.retire-20261030
及/tmp/gef-history-migration-20261030内的临时源码、环境、构建和原始日志；所有输入对应服务均已终态。
随后更新最终清理凭据并正常推送当前分支，不再改写历史。SSH经现行netpath的认证HTTP CONNECT已验证，
临时代理helper只在进程内读用户认证，不含凭据，不改公共配置；清理后不作为开发依赖。
后续开发从[development](development.md)重建环境；旧号通过[history-migration](history-migration.md)查源码。
原始性能任务不恢复，当前限制与停止/重启条件只在[ROADMAP](ROADMAP.md)。
