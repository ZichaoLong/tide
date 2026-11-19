# 小目录收尾（2026-11-19）

本次按用户明确的 **整个仓库含Git和私有环境低于200,000,000 B** 目标处理。
收尾观察为 **144,379,904 B（144.380 MB，
137.691 MiB）**；后续文档/Git提交只增加少量块。
本地工作完成，推送因SSH网络超时待恢复；不强推，不改共享路由或登录凭据。
详细记录见[审阅JSON](compact-repository-20261119.json)和
[冻结输入、过程记录及失败日志](compact-repository-20261119-records.tar.gz)。

## 留下什么

| 范围 | allocated MB |
|---|---:|
| 共享 `.git`（历史、原实验分支及下游worktree元数据） | 109.744 |
| 文档及压缩审阅证据 | 10.691 |
| 主私有环境 `.venv-instances` | 13.537 |
| 本次小型维护脚本/收据 | 0.397 |
| 源码、固定数学、测试、配置等其余部分 | 10.011 |

426个旧artifact根、19,994个文件，合计2,200,645,632 allocated B已经退役；
其中包括上一阶段保留的全部张量、局部见证、失败profiler、原始结果gzip和重复私有安装。
18个实验worktree通过 `git worktree remove` 移除，所有原实验分支refs保留；
注册worktree只剩主仓和下游。没有将大文件移到别处计作释放。
后续旧路径是测量身份，不是在线可用性保证。历史失败仍为失败。

## 源码与证据保存

历史保存提交 **`9b270280227f22190510767da50d643a049daa49`** 包含：

- `retired-inputs/worktrees/<name>/`：18个tracked树，树SHA逐一等于原worktree HEAD的树。
- `retired-inputs/ignored/artifacts/...`：6,171份源码、配置、输入身份、小JSON/XML结果；
  覆盖独有native版本、restore脚本/笔记、development patch及所有被审计的独有小输入。
- `retired-inputs/manifest.json`：原HEAD、树和文件/blob/SHA映射。
- `retired-inputs/job-records.json.gz`：855份历史launch/status原值；不把旧failed或stale running改成passed。

下一个提交 `077b583d` 恢复原主工作树内容，使这些历史副本不常驻。
这是保存历史树，不是采用这些分支的模型行为。当前生产 `src/tide` 与 **`97c52960`** 相同。
本地Git保有完整保存提交；网络恢复后的普通当前分支push会携带这段历史。

47份审阅JSON的原字节由67,962,999 B压缩到
6,594,104 B，保存在 `docs/evidence/*.json.gz` 并纳入Git。
全部重新打开磁盘gzip完整解压，与原Git文件逐字节一致；原阈值、失败、source与数值均未重写。
53处导航链接和三段读取入口已迁移；26个退役artifact链接改为明确的历史路径文本。

当前[归档索引](archive-index.json)指向这47份保留证据。旧
`storage-archive-index-20261119.json.gz` 的352条artifact索引只保存历史身份，
不能继续据此还原已退役的原始报告。使用固定 `bef26abb` 的新入口：

```sh
python scripts/materialize_record.py \
  --index docs/evidence/archive-index.json \
  --record docs/evidence/memory-candidates-20261119.json \
  --output /tmp/tide-memory-review.json
```

输出必须尚不存在；工具核验双SHA、长度和gzip完整性并原子发布。
查看历史源码时按需使用 `git show 9b270280:retired-inputs/manifest.json` 或
`git archive 9b270280 retired-inputs/worktrees/NAME`，输出到新位置。
这些操作不能恢复已退役的模型历史状态；进一步数值调查需另选有限输入，不保证旧整图轨迹重现。

## 核验、边界与异常

还原工具10项检查通过；最大审阅JSON的真实还原通过；47个磁盘gzip均完整验证。
清理后从 `97c52960` 构建新隔离wheel，23包文件与源码一致。
仓库外 `python -I` 的CPU FP32公开入口（FC n8/layers4/d64、B2/L8）完成训练；
checkpoint数据和下一前向exact，实际配对更新在原阈值内；
另一新解释器恢复后position8→16并真实更新。检查不依赖旧artifacts。
结果先提交，再退役本次61,870,080 B临时验证输出和冗余本地元数据。
这不新增9B训练、NPU/CUDA性能或模型质量资格。

固定math的108文件及四份用户未跟踪文档原字节不变；未修改下游文件、公共环境和参考仓库。
下游在清理期间出现并发未跟踪文件，本次未写入这些路径。删除后的独立检查确认
当时下游HEAD与tracked树不变；不把它的并发工作清除来满足状态相等。

保留的执行异常也属于收尾证据：第一次删除预检使用了本机Git不支持的ls-tree格式选项，
在任何删除前失败，随后改用兼容输出解析。第二次已完成全部删除，但最后的下游status
相等断言因并发新增文件而失败；独立事后检查通过，原服务退出1仍保留。
SSH直连、现有c28/cdn-vps的有界推送也超时，记录为推送失败，不冒充已发布。

7个 `background.slice` 服务、19个记录后代全部终态，服务时长合计约142.788秒；
最多8个CPU/16 GiB，CPU安装检查最多2个CPU/4 GiB，无NPU任务。
没有执行额外Git repack；Git普通提交过程可能自动打包。检测到下游并发工作且目标已达到后，
未额外改动共享对象库。原实验refs与主/下游worktree均保留有效身份。
账户quota与逐项尺寸在配套JSON；这里的服务时间不是模型吞吐。
