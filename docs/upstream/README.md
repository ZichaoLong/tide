# 固定上游参考快照

`tide-core-3/` 的五份 Markdown 是 [upstream.json](../upstream.json) 指定提交的
逐字节副本。数学权威仍是原 ObsidianVault 提交；这里供离线导航与核验，
不单独修改、不形成第二份数学规范。文内可选扩展链接仍可能指向原上游材料。

```sh
python scripts/check_upstream.py
python scripts/check_upstream.py --upstream-repo /path/to/ObsidianVault
```

第一条只核验仓库快照；第二条还与指定上游仓库的固定 Git 对象逐字节比较，
不读取其工作区草稿，也不切换分支。均不需要 Torch、设备或历史实验文件。
更新上游时先比较定义与实现影响，再更新锁定提交、快照和校验值；
不能只替换版本号。旧快照可由本仓库 Git 历史取得。
