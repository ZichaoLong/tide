# 固定上游参考快照

`tide-core-5/` 保存 `0770593838a37c790a568226930567ed60f99b83` 的 19 份正典、
局部定义、拓扑、实例、采用记录及验收条目，逐字节来自固定 Git 对象。
[upstream.json](../upstream.json) 是唯一来源与 SHA256 清单。
`tide-core-3/` 仍保留旧 profile/历史引用所需的五份原文，锁在同文件的 `legacy` 条目；
它不代表当前新支线的采用版本。快照均不就地修改。

```sh
python scripts/check_upstream.py
python scripts/check_upstream.py --upstream-repo /path/to/ObsidianVault
```

第一条离线核验所有快照，第二条额外与各自固定提交的 Git 对象比较，均不读取
上游工作区内容。嵌套路径、额外/缺失文件、重复清单、哈希修改与符号链接均拒绝。
原文的可选扩展链接保留上游形式；实现对应入口为 [semantic-map](../semantic-map.md)。
