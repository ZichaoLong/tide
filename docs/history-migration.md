# 历史清理与提交号追溯

2026-10-30按用户授权，后续基座仍叫`graph-execution-foundation`。
原分支在`83d384a4a5efb9df38ce77c395c2d9c5222b4e0e`冻结为
`graph-execution-foundation-old`，保留在同一远端。旧分支不继续开发，也不合并回新分支。

新历史仅移除`docs/evidence/archive/retired-records-20261012.jsonl.gz`。
716个代码提交及父链全部保留，其中181个提交号改变；作者、提交时间、消息和其他文件原样保留。
过滤后的原HEAD为`0cc8811af02e43ef1365b4f5a4cc797575196438`；其后的提交交付本页、工具和证据。
完整参数、原归档hash及映射hash见[迁移清单](history/migration-20261030.json)。
这不是丢弃代码历史的单提交基座，也不修改固定数学上游。

## 新检出与旧工作区

```sh
git clone --single-branch --branch graph-execution-foundation \
  git@github.com:ZichaoLong/tide.git
```

此命令保留新分支的完整代码历史，并限制日常fetch只跟踪新分支。
普通全分支clone仍会下载`-old`中的大归档；同一Git对象库保留旧引用也不能获得同等缩减。
旧工作区应先保存未提交改动与未发布提交，再另建上述新检出；不要直接pull/merge两个历史。
需要移植私有修改时审阅补丁并在新基座重新验证，不能按新旧提交号不同就认定代码不同。
远端同时保留旧历史，所以不承诺整个GitHub仓库的总存储缩减；本轮缩减的是新分支历史与日常检出。

## 从旧证据定位源码

```sh
python scripts/resolve_history_commit.py 35d6358
python scripts/resolve_history_commit.py 83d384a --json
```

工具接受旧/新完整SHA或至少7位的无歧义前缀，也接受迁移后新增的本地提交。
它核验[完整映射](history/commit-map-20261030.tsv)的SHA256和目标对象，未知、歧义或浅克隆缺失均失败；
不联网，不导入旧分支，不将数学上游或LH的hash解释为本仓库提交。
可将输出交给`git show`/`git checkout --detach`查看新历史中的等价代码。

当前契约与导航里的受影响提交号已指向新历史。历史报告、JSON凭据、二进制hash、
运行记录及`.torch-portability/contract.json`保留当时实际身份，经本映射定位源码。
原提交消息中的hash也保留原文。映射证明归档之外的源码相同，不证明历史二进制已经在新提交上重新运行。
历史绝对路径和文件hash仍只是身份，不能恢复已退役的原始运行材料。

下游已有依赖锁和checkpoint不自动改写。旧安装物继续按原身份使用；升级采用新提交、
重新构建/安装与接入验收，并保存新锁，见[协作规范](downstream-contract.md)。

## 按需取得旧归档

[固定原提交上的归档](https://github.com/ZichaoLong/tide/blob/83d384a4a5efb9df38ce77c395c2d9c5222b4e0e/docs/evidence/archive/retired-records-20261012.jsonl.gz)
仍有原始40,407,191字节；SHA256为`8d5a564fcc6cbb80dada6f978585931284ed3972020bde6f9eb393ea6c4d7194`。
必要时从该地址另存到仓库外，或单独克隆旧分支。不要为一次查询把旧引用fetch进日常检出。

```sh
python scripts/read_retired_record.py --archive /path/to/retired-records-20261012.jsonl.gz \
  --list-prefix /mi/data2T/zlong/tide-accelerator/sources/native-dev01
```

未提供实际归档时明确失败，不自动下载。整个归档和提取记录仍校验hash、拒绝覆盖已有输出；
旧脚本不会被执行。该归档的历史保存范围见[产物规则](artifact-retention.md)。

## 独立验证与维护

两个独立检出分别提供原对象和新对象后，可以重做逐提交核对：

```sh
python scripts/verify_history_mapping.py --old-repo /path/to/old-clone \
  --new-repo /path/to/new-clone --output /path/to/new-verification.json
```

验证器不调用过滤工具，逐一比较Git树中的路径、模式、blob和有序父链，以及完整提交元数据/消息；
只允许指定归档消失，并检查原归档字节身份及新历史不可达。另用独立合成分叉/合并历史测试
非归档内容、文件模式、父链、消息、映射损坏/遗漏、歧义前缀和缺失目标的拒绝。

映射和清单是不可追加的本次迁移凭据。以后若再改写历史，应交付新的显式迁移及映射，
保留既有链路。普通开发只更新[STATUS](STATUS.md)、相关契约和证据，不需要重写本清单。
