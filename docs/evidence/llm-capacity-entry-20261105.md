# 显式设备容量试跑入口（2026-11-05）

实现固定`59e9acad6c5c3c7b106c5882acc6e3e8bd58c678`，非editable wheel已安装到项目私有环境。
135项设施/随机微型CPU检查通过，0失败/0跳过；开发37项与之重叠，不累计。
已验安装后的plan和queue --config帮助、原15项清单及配置身份不变。完整[小型证据](llm-capacity-entry-20261105.json)
含wheel/package hash、Junit身份、固定检出、公开配置及设备队列helper身份。
没有启动预训练B512、NPU模型或新的性能测量；本报告不证明它能完成或给出峰值。

## 公开行为

- `execution.device_memory_admission=observe`只对显式加速器Transformers推理放行静态显存预算。
  原estimate/free/total/reserve均先落盘，超限写observe_exceeded；默认strict仍拒绝。
- CPU主存/磁盘、资产、设备可用性和数量、空闲卡组锁、时间上限、OOM/非有限/运行异常保持原处理。
  模拟OOM检查验证准入记录保留、结果failed、异常上抛；没有吞掉错误换成completed。
- `lm queue --config`将单配置展开为有身份的单项清单，复用原监督器；与--manifest互斥。
  终端/CSV/HTML显示memory admission，不能把容量探索当原strict配置通过。
- 新配置保留原双NPU、Qwen3-30B-A3B-Instruct-2507、BF16、B512/L512/O128、prefill64、16线程、
  一次warmup/正式测量；只新增observe身份，并把整流程计算上限明确设为7200秒。

资格由background.slice的`gef-lm-qualify-20261105-i.service`执行，1800秒上限。
终态MainPID=0、ControlGroup为空、原supervisor/workload PID/start均无匹配存活进程；固定检出仍干净。
原模型失败及容量拒绝见[15项终态](llm-isolation-terminal-20261105.md)，未改写。

## 运行入口

本机固定源码：`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102/source/capacity-trial-i`。
现有环境/资产根：`/mi/data2T/zlong/home/llm/gef-lm-experiments/20261102`。
[公开配置](../../examples/external_lm/qwenmoe-two-npu-b512-observe.json)的config_id为
`be83db482b3ee3e182ad5832c0c491d94dddf7e7a06613686055ddab736e45a8`。
启动、排队状态、日志、结果和完整停止命令见[操作说明](../external-lm.md#显式容量试跑与单配置后台运行)。
用户执行最后提交命令后才会创建B512运行；本次没有预先提交它。
干净仓库可按操作说明安装同提交、准备固定资产并重建，不依赖已完成的旧模型输出。
