# 当前交接 — 2026-10-31

当前任务：core-5 CUDA 真机开发与资格；授权分支 `graph-execution-foundation`，允许阶段提交和普通推送。
启动 HEAD `a066756fe23260e45d61afa158f87883d426f0d5`，工作树干净，无旧活动作业。
数学固定 `0770593838a37c790a568226930567ed60f99b83`，19 份 core-5 快照及旧 core-3 快照校验通过。
历史治理已完成；只取新分支，原证据身份/映射见 [history-migration](history-migration.md)。

## 现场与范围

- 经本机认证 HTTP CONNECT 连接 `ssh gpu3090`；凭据只在私有配置/内存，无全局配置修改。
- 复查：x86_64、Python3.12.3/Torch2.12.1+cu130、Toolkit13.0.88、GCC11.4、CMake3.22.1、Nsight2025.3.1。
- 四张 RTX3090/24GiB/SM86 均空闲；12方向 Torch P2P 查询全部 false。多卡 resident 硬件条件不足，其他路径继续。
- CPU affinity112、配额56核；内存240GiB、当前约1.1GiB；数据盘50GiB基本空闲。初始构建56线程，显式sm_86。
- 非登录 SSH 没有 `python3` PATH；使用 `/root/miniconda3/bin/python`。首次探测因此退出127，随后绝对路径探测通过。
- GPU是主线，先单卡FP32完整训练/推理/恢复，三客户端分别验；CPU仅小型公式与相关回归。
  resident/FP16/可用多卡及有限性能随后；不提高原容差，不把复制成功称直接P2P。

## 当前状态与下一步

目标根 `/root/autodl-tmp/gef-cuda3090`；私有 `venv` 使用原Torch2.12.1+cu130，pytest8.4.2/Ninja1.13.0。
CUDA13 API修复 `1fe1e37`、peer/IEEE记录 `31d0afd`、独立失败继续门禁 `c368b45` 均已提交推送。
固定c962961 core构建209目标通过；1fe1e37 resident standalone630/Python255目标通过，56线程/sm86。

已终态：
- FP32：PyTorch19/19、Python native19/19；standalone23/24。KV mixed-a/b三客户端6/6。
- standalone fc/ssd FP32在第3步AdamW后regions.0.head.bias失败，差3.4869e-6超原容差；第3步梯度CPU -1.47309e-7/GPU -1.57744e-7。GPU scalar仍失败，CPU FP32/FP64通过；mixed-a另见恢复后gradient单ulp不等。无容差/输入变化。
- FP16：PyTorch19/19、native19/19、standalone22/24；feedback/affine、feedback/ssd严格tensor失败待字段诊断。
- `jobs/peer-31d0afd` 72单元通过；12方向P2P/原子均不支持、enable217。AccumulateGrad流警告保留。
- `jobs/multicard-31d0afd` failed：传输根测试、standalone generic36案例/72更新/144连接窗口通过；五记忆+异构PyTorch6/6、native5/6；显式Python CPU桥delta-pdg完整通过。native gated-hb在同栈第四步梯度3/4坐标逐值不等，最大4.44e-16。不称直接P2P/NCCL/resident资格。
- 最初两次resident构建86-real配置/CUDA13 API失败及`resident-single-1fe1e37` control末尾workspace断言失败均保留。

`jobs/resident-inventory-c368b45` 已failed/exit1且无存活后代；56组件中54通过，物理3、10800秒上限，复用源码hash匹配1fe1e37库，
`runs/c368b45-resident-inventory` 会保留全部组件及公共文件的独立失败；完整门禁不允许跳过。
旧公共Python53项通过且零跳过；整体因组件失败仍failed。control workspace和recurrent FP32 one-byte预算测试失败。开发已修复：CUDA纯控制用实际retained operands/metadata核验，
CANN原workspace断言保留；组件预算改为扣除既有CUDA error flag后的增量，原one-byte不足拒绝保持。
修订control8轮、recurrent FP32/FP16各24配置目标开发通过；待固定提交重建/资格。

6cacad0已交付native稳定传输顺序修复：TensorImpl只识别别名，packing依字段首次出现顺序，避免新owner地址决定VJP边顺序。
反例在旧实现失败，修复后C++ CPU FP32/FP64和相关placement 110项通过；原双卡gated-hb精确恢复仍以相同3坐标/4.44e-16失败，不能称已修复该边界。稳定字段次序修复单独交付；停止重复相同恢复候选。
新增standalone独立根、scalar/peer资格及原参数梯度诊断已定向CPU FP64、双卡CUDA KV通过。
standalone benchmark发现干净检出缺失已退役LH vendor JSON目录；改为显式外部nlohmann头依赖和完整hash，私有环境补3.11.2及importlib_resources7.1.0，不恢复旧目录；C++ IEEE政策显式记录。

修复实现已分三次提交推送：6cacad0（稳定顺序/独立根/peer/scalar），8a34f66（控制与增量预算），32edb0f（外部JSON头/IEEE）。
`jobs/build-all-32edb0f` 已passed/exit0且无存活后代：冻结32edb0f，core→resident standalone/Python→standalone benchmark，56线程/3600秒；新目录 `builds/32edb0f-*`。
四阶段构建均通过，56线程、显式sm86；source/binary/外部JSON头hash各由manifest记录。
`jobs/resident-single-32edb0f` 已passed/exit0：冻结32edb0f、物理3、10800秒，使用三个32edb0f匹配构建，56组件/dtype与53公共项零跳过通过，输出`runs/32edb0f-resident-single`。
`jobs/eager-regression-32edb0f` 已failed/exit1且无存活后代：同固定来源、物理0/1、7200秒，相关CPU回归、两dtype standalone独立根/严格恢复、scalar及双卡各族；输出`runs/32edb0f-eager-regression`。CPP两dtype原3例及native peer GDN失败保留；CPU110+两dtype核、scalar18、CPP双卡6均通过。
`jobs/profiles-32edb0f` 已passed/exit0，物理0、3600秒上限，输出`runs/32edb0f-profiles`：控制与两resident客户端Nsight node时间线、两eager客户端Torch profile。全部资格/构建已终态，profile不作为吞吐。两个resident消费者profile内完整核对/精确恢复通过，图内kernels与host graph launches分列。
当前 `jobs/finite-32edb0f` running，物理0、10800秒，串行37个显式选择，输出`runs/32edb0f-finite`；3个测量窗+1warmup/配置，完整记录/梯度/SGD/精确恢复决定吞吐资格。
`jobs/half-boundary-32edb0f` 已failed：两项原FP16 feedback失败在CPU复现完全相同参数/值，不能归因为CUDA专有缺陷。
`jobs/half-packing-32edb0f` 已failed且无存活后代：显式scalar SSD在CPU/CUDA各通过，affine仍相同失败；原packed两例均保留。无生产精度/容差变化。
有限目录a864fef已提交推送；6个模型静态展开/执行命令校验通过，最大约345万参数。所有资格终态后运行`profile-32edb0f.py`和显式37项`finite-32edb0f.py`，均在目标根，source32edb0f。
再对固定提交运行相关单/双卡严格恢复、单卡resident及timeline；无关已通过范围不重复全矩阵。
全部正确性/重型构建终态后再开始有界profile与小中规模FC/HB/LH对照，失败案例不报告合格吞吐。
审阅证据单独提交；旧CPU/NPU严格失败不自动解除。未提交文件见git status；首轮eager审阅证据与能力声明独立于新来源资格。

所有目标日志为 `jobs/NAME/task.log`、`status.json`；输出为 `runs/NAME`，从未覆盖旧运行。
活动源码只读；`dev` 为dirty开发，不能作为正式来源。无本机长作业。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/venv/bin/python /root/autodl-tmp/gef-cuda3090/sources/c368b45/scripts/container_job.py status --output-dir /root/autodl-tmp/gef-cuda3090/jobs/resident-inventory-c368b45'
```
