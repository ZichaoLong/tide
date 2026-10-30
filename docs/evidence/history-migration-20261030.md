# 历史迁移与新CPU接入验证 — 2026-10-30

固定实现`b0969fc9819e30b9e36fad52bd2d4484dff16206`；数学仍为`0770593838a37c790a568226930567ed60f99b83`。
[机器凭据](history-migration-20261030.json)保存命令、终态、源码、环境、binary hash和检查摘要。
[迁移契约](../history-migration.md)提供旧→新映射、单分支clone与旧归档取回。
远端发布及最终本地切换在发生后另记；本报告只声明已实际完成的过滤与固定源码验证。

## 历史完整性

仅删除新历史中的40,407,191字节归档；代码历史716个提交全部保留，535个原号、181个改号。
独立验证器直接读取两个对象库，不调用filter-repo：全部非归档路径/模式/blob、有序父链、
作者/提交者/时间及原消息逐一相同。原归档byte/hash符合固定凭据，在新历史不可达。
该验证不执行旧实验，不改写历史实测身份。当前导航改用新源码位置，旧报告通过映射追溯。

## 固定源码资格

独立服务`gef-history-qualify-20261030.service`于13:54:37Z启动、13:59:32Z以0退出；
实际源码干净。aarch64、Python3.12.13、Torch2.12.0+cpu、现行module和私有增量环境。
320个可用CPU，构建使用half=160；测试ATen/BLAS线程1，background.slice/Nice10，
45分钟/512GiB/16000%CPU上限。没有修改公共工具链，没有执行NPU/GPU。

| 范围 | 实际结果 |
| --- | --- |
| Git历史独立核对 | 716提交；181次归档出现均移除，其余内容/元数据/父链相同 |
| 固定教材快照 | core-5的19份、保留core-3的5份通过 |
| 新CPU构建及算子doctor | 通过；全新build，源码和二进制hash保存 |
| CTest | 13项通过 |
| standalone smoke | CPU FP32/FP64前向与反向通过 |
| 定向pytest | 177通过、无失败/跳过；含映射/归档/持久记录、语义、native Settle和core-5 recipe训练 |
| 仓库外安装消费者 | 4项单元、8组独立记录/训练/新进程精确恢复、4项非法请求拒绝通过 |

开发期32项检查已包含在177中，不累计为另一批资格。52条pytest warning来自既有recipe的
record_property与xunit2报告格式，不是失败；原始数量保留。安装消费者覆盖FC的两客户端×两精度、
HB mixed两客户端FP64、FC streaming/scalar两客户端FP64，其余为prefill/packed。
该定向检查不代替全后端/全配置/性能矩阵，也不改变ROADMAP中数值、异步和硬件停止条件。

## 重现与保留

在新分支完整clone中检出固定实现，按[development](../development.md)创建私有环境。
机器凭据的command_script给出准确门禁；替换临时根，另提供冻结旧分支clone给历史验证器。
已有依赖锁与checkpoint不自动改号；新消费者捕获新构建身份并完成实际安装验证。
旧绝对路径和二进制hash是这次运行身份，不要求永久保留临时环境、构建、日志或checkpoint。
