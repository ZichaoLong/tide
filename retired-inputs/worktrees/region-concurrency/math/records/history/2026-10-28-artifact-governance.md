# 2026-10-28：生成物治理与最近历史清理

按用户明确同意的方案，Git 跟踪维护源、原始素材与留存证据；可重建的展示产物默认不跟踪。当前规则见[仓库约定](../../../ARTIFACTS.md)，准确路径与构建命令见[生成物清单](../../../generated-artifacts.json)。

## 历史范围与引用

保留 `0770593838a37c790a568226930567ed60f99b83`（2026-10-27 11:52，Asia/Shanghai）及全部祖先。该提交仍是下游实现固定采用的数学版本。从其后的七次提交中移除 45 个图集与展示产物，保留提交顺序、作者、提交者、原时间和说明。正文及源码仅替换受影响的提交编号引用；完整旧讨论稿仍保存在对应的新提交中。

| 原提交 | 重写后提交 |
|---|---|
| `c0afcff` | [`6517ac5`](https://github.com/ZichaoLong/ObsidianVault/commit/6517ac525bce57bd20ea8f22753407d2fc331982) |
| `d9aa499` | [`a8377d5`](https://github.com/ZichaoLong/ObsidianVault/commit/a8377d5916057c8c6d45f864919ceb5c36fc178c) |
| `948ee3f` | [`9c9b870`](https://github.com/ZichaoLong/ObsidianVault/commit/9c9b8700d56ef831d6f1021346f877e0b5fe1a07) |
| `303be9b` | [`d15bafe`](https://github.com/ZichaoLong/ObsidianVault/commit/d15bafe9df9daad0f77b5eb319651e2d15f5e2ae) |
| `696e405` | [`9cd9f11`](https://github.com/ZichaoLong/ObsidianVault/commit/9cd9f111a6ad6ece0f0e12962c5573e112d70ec0) |
| `6d313f6` | [`796c6cf`](https://github.com/ZichaoLong/ObsidianVault/commit/796c6cf19153fc679b218b6e25f5ccf991e6d051) |
| `702590a` | [`6f99104`](https://github.com/ZichaoLong/ObsidianVault/commit/6f9910479b9df8e2755dde940633fff4342664a4) |

[完整映射与路径清单](artifact-rewrite-map.json)保存原编号供核对。历史记录中对这些提交的链接与来源版本已映射到新编号；历史核验中的 `0770593` 和外部实现提交编号保持。字符串形式的旧编号不会使旧 Git 对象继续被引用。

四份正典、全部局部教材、拓扑和实例在这段期间没有修改。逐提交验证重写前后的文件集合和内容：除登记的生成物和提交引用外，其余文件对象逐一相同。

## 当前维护方式

- 生成物保留原输出位置，使用精确忽略规则；图集和展示入口说明先构建再打开产物链接。
- 检查器允许未生成的已登记产物及其明确输出目录，其他失效链接继续报错。完整构建后使用 `--require-built` 检查全部输出。
- 新增日常 `local-render-checks.json`，与此前留存的 `local-provenance.json` 分开；已有来源与实现核验快照保持留存。
- 手工 SVG 可转为独立源文件，构建器读取该源文件并避免覆盖。缓存和浏览器临时文件忽略，其他原始素材与未归类笔记保留。

## 备份与已有克隆

执行前完整备份了 Git 数据库和工作区（含未跟踪文件），位置为 `/home/zlong/.local/state/obsidianvault-governance/2026-10-28-artifacts/original-repository-and-worktree.tar.gz`；同目录保存 SHA-256、执行前状态和逐提交核验结果。备份位于仓库外，不随 Git 推送。

远端更新使用原 `master` 编号作为 `--force-with-lease` 条件，避免覆盖执行期间其他人新推送的提交。已有克隆应先保存本地工作，再重新克隆或将本地分支对齐新历史；不要把旧提交链合并回新分支。保留的 `0770593` 及以前版本可照常查询。

停止跟踪本身不释放历史对象。完成远端更新后，本地回收不再可达的旧对象；远端的物理回收时间由托管服务决定。
