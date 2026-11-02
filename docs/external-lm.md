# 外部语言模型实验入口

本入口测量外部预训练语言模型，独立于core-5图执行器及其资格。
固定模型、有限计划和预算见[准备报告](evidence/llm-experiment-preparation-20261102.md)。

## 已实现：固定资产

安装后的入口是`tidegraph`，Python要求>=3.10。

```sh
tidegraph lm catalog
tidegraph lm fetch qwen05 --root /absolute/experiment/models --project-root /absolute/experiment --project-gib 80
```

`catalog`和`fetch`不导入Torch。下载调用遵循启动环境的HTTP(S)代理；本机按netpath技能从已有c28入口执行。
每个文件按固定大小和官方SHA256或Git blob身份校验。部分文件以`.part`保留；明确重启相同命令才续传，不自动重试失败网络。
`asset-state.json`的complete仅表示所有资产校验完成，不表示模型加载、推理或训练通过。

长任务使用`scripts/submit_lm_job.py`和固定干净源码；该入口复用已安装的持久子进程监督器。
项目长任务目录置于主仓库外，用户主工作区后续修改不会进入活动任务。
主环境复用只读公共Torch2.12/CANN9.1，`scripts/prepare_lm_environment.sh`只写项目私有overlay。
运行配置、模型执行、数值检查和报告适配仍在后续实施阶段，不能将资产入口当完整LM运行入口。
