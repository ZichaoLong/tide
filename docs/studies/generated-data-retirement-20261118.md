# 公开数据单独整理与生成数据清理：2026-11-18

用户要求将需要下载的公开数据单独拎出，后续由用户移入公共目录；其他可重新生成、计算的数据删除。
本次已完成：`runs/`为空，删除16个生成文件和6个内部符号链接，实际释放821.883MiB。
数据清理后仓库约429MiB；后续环境清理后降至约103MiB，见下文。
另有299.527MiB原始语料移到仓库外，未删除，因此仍计入账号占用。
本次没有运行训练、数据重建或模型评价，也没有改动其他项目及共享环境。
[执行与校验凭据](../evidence/generated-data-retirement-20261118.json)。

## 公开数据交接

原先按中性、可扩展布局准备在`~/public/`，用户现已迁入`/mi/data2T/public/huggingface/`。
本次已将其中7个目录设为2775、5个普通文件设为0664，目录带setgid以便继承所属组。
文件内容核验不变；所有者/组未修改，用户后续统一`chown -R root:devusers`。
[权限设置凭据](../evidence/public-data-permissions-20261118.json)。当前布局：

```text
/mi/data2T/public/
  huggingface/
    README.md
    datasets/Salesforce/wikitext/revisions/
      b08601e04326c79dfdd32d625aee71d232d685c3/
        manifest.json
        SHA256SUMS
        wikitext-103-raw-v1/
          train-00000-of-00002.parquet
          train-00001-of-00002.parquet
```

目录和附带元数据中没有TIDE、账户名或本机home路径。原带项目名的暂存目录已移除。
后续同类下载沿用`<来源>/<资源类型>/<命名空间>/<仓库>/revisions/<固定版本>/`，
每个版本清单只列实际下载的原始文件；新版本另设目录。
只含公开的`Salesforce/wikitext`原始train两分片，固定版本
`b08601e04326c79dfdd32d625aee71d232d685c3`、子集`wikitext-103-raw-v1`。
权限操作仅涉及`huggingface`子树，未改父目录`/mi/data2T/public`的权限、所有者/组或其他公共资源。

每个版本目录包含`manifest.json`和`SHA256SUMS`，`huggingface/`内含通用`README.md`，记录下载URL、版本、字节数、SHA-256及上游
CC-BY-SA-3.0/GFDL许可声明。两份原始文件移动前后均与旧数据锁中的SHA-256相符，
独立执行`sha256sum -c SHA256SUMS`也通过。移动使用同文件系统rename，没有再复制300MiB。
[新布局核验](../evidence/public-data-layout-20261118.json)保留实际相对路径和文件身份。

删除的内容包括展开文本、训练/验证/确认/test token数组、各轮完整数据锁与划分记录及准备状态。
数据来源与文件hash保留在小型清理凭据和暂存清单中；这不能替代原来的完整划分与排除列表。
后续必须重新准备并核验数据身份，不把重新生成的划分自动当成旧确认集。旧权重已在第一阶段删除。

## 三个私有环境与整个.local已删除

|目录|实占MiB|用途|本次处理|
|---|---:|---|---|
|`.venv`|166.0|旧基座85d161d、早期A/B实验、私有Transformers/tokenizers等|后续明确授权后已删除|
|`.venv-positive-floor-v1`|17.0|基座74e7bbd的positive-floor-v1安装和限定验收|后续明确授权后已删除|
|`.local/language-data-venv`|142.2|pyarrow20.0.0，用于Parquet预处理|随整个.local已删除|

它们是Python隔离环境，均通过system-site-packages复用`/opt/miniconda`中的公共基线，
并非三份完整PyTorch/NPU安装。删除这些本项目目录不会卸载公共环境或删除公开模型。
数据清理时仅检查环境。用户随后明确批准删除两个`.venv`和整个项目`.local`，现已完成，
合计另释放325.922MiB；三个Lens writer注册在删除前已通过官方CLI注销。
`.local/cleanup-20261118`约756KiB，内容为第一轮清理的一次性执行器、固定清单、日志及原始回执，
任务结束后不再需要，已一并删除；主要清理结果及失败说明仍在Git中。
私有`.local/LENS.md`也按授权删除，其内容未写入公开记录。
[环境及.local删除凭据](../evidence/private-environment-retirement-20261118.json)记录范围和核验。
上述项目内环境清理未涉及账号级`~/.local`、共享`/opt`环境或Lens服务/访问策略。
后续单独授权的账号级清理见下一节。

源码、配置、依赖身份、安装wheel和审阅结论继续保留；不承诺公共基线或包源未来不变。
旧安装锁是当时的记录，环境重建后须重新确认实际安装与来源。

## 后续授权的账号级.local首批清理

只读调查后，用户同意优先推荐的首批范围，现已删除以下6个目录，按实际分配块去重释放6.563GiB：

- `~/.local/state/trackio-lens/packaged-skill`
- `~/.local/state/trackio-lens/managed-acceptance`
- `~/.local/state/trackio-lens/managed-acceptance-repeat`
- `~/.local/state/trackio-lens/fresh-install-check`
- `~/.local/state/trackio-lens/compatibility-tools`
- `~/.local/share/focus/_global/web-updates/2edf24dec15f2561477889843dbb5623`

旧SDK测试解释器先经官方CLI注销writer注册。FOCUS升级状态为succeeded，升级作业已退出；
当前安装是独立副本。Lens/FOCUS/netpath的主PID和启动身份、注销后的Lens配置均未改变。
当前正式服务安装、Node/Codex、公共数据、Ascend和netpath的第二批维护候选保留。

第一次预检将所有硬链接一概拒绝，未删除任何目标。定位到8个esbuild inode的两份路径均在
各自固定删除目录内后，新固定r2检查完整的目录内链接计数并去重统计，随后完成删除；
没有放行跨删除边界的硬链接。两次工作均在background.slice，2核/768MiB/每次600秒上限。
同账户进程巡检有47个FD权限不可读的进程，这一可见性限制仍记录；候选中未见可访问的活动使用者。

账号级`.local`可读取占用约10.25→3.68GiB。普通目录扫描仍有4处容器层权限不可读，
该值不是跨权限的完整统计。临时执行器、日志和回执在确认服务结束并写入
[小型清理凭据](../evidence/account-local-cleanup-20261118.json)后移除，没有新建持久的cleanup目录。
