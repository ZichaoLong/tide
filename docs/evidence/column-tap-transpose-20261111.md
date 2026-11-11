# 固定时间卷积的有序反向与NPU精确恢复

实现固定`8c72ca945c222af2173e3fafd52f104ee9f12034`；[身份、配置、运行摘要](column-tap-transpose-20261111.json)。
原b993f38的SSD/GDN/混合LH精确恢复失败保留在[公共消费者报告](experiment-columns-20261111.md)。

## 原因与修复

卷积查询同一cache行时，不同lag可以重用它。普通index_select反向的碰撞浮点累加在本NPU上
不逐位确定；同一checkpoint重新建两个owner也出现参数/optimizer尾差。诊断用全局deterministic
开关能消除该见证，但没有把全局开关作为基座修复。

新`tap_gather` / `columns::gather_taps`利用严格递增原始查询时间：同一lag内真实cache行至多出现一次。
host int64逆表按`bank row × lag`收集cotangent，再沿lag轴归约；前向仍是整列gather，
缺失槽是零sentinel。保存原values版本守卫，明确仅一阶VJP，区分未使用与连接零。
对应教材固定时间卷积的同一线性转置，没有压紧稀疏时间或改变窗口。
新增bank×lag反向暂存和host/device int64索引计入容量，包含非空旧cache。

## 固定资格

- CPU定向74 passed。独立关联矩阵、方向导数、缺失槽、空bank、None/零、阶数与原地版本守卫；
  native完整五记忆/快照/共享/非空左态/训练及旧cache容量拒绝。开发16项与后继74项不累计成新覆盖。
- Python-owned NPU与独立SDK分别真实构建。固定源码与开发构建全部C++字节及二进制hash逐项核对复用；
  原manifest不改写。standalone CPU五记忆与独立streaming、raw状态/VJP、两SGD窗通过。
- 空闲NPU物理11，CANN9.2/Torch2.12：SSD、GDN、176节点混合LH×python-native/PyTorch，
  每个从两窗训练后的同一checkpoint启动两个新进程；实际输出/loss、raw状态、参数、梯度、optimizer与有效配置
  **全部逐位相等**。这六个检查明确记录`deterministic_algorithms=false`；合法tap重复VJP也全相等。
- 四个安装消费者（GDN/SSD×Python/native）独立三步AdamW数值及新进程恢复通过。
  **这些旧示例沿用deterministic_algorithms=true**，不能用它们单独证明默认算法的重复性；上项六个检查提供该证据。

本轮修复闭合具名的六个配置，不外推所有后端/规模、异步多卡或其他含重复索引的算法。
原始服务column-consumer-npu仍failed；新的column-tap-npu为passed，不能重写旧终态。
该修复没有新增性能声明。新旧数学相同，但安装/源码身份不同，不认证跨版本精确续接。

## 可重现入口

使用固定提交与匹配构建，从[开发入口](../development.md)安装后：

```sh
TIDE_BUILD_DIR=BUILD python -m pytest -q tests/test_tap_gather.py tests/test_native_recurrent_columns.py tests/test_recurrent_columns.py tests/test_column_training.py tests/test_experiment_columns.py --dtype both
python scripts/check_column_repeatability.py --device npu:0 --output-dir NEW_GATHER
python scripts/prefill_regression_config.py --case layer-chain --family ssd --device npu:0 --threads 2 --output NEW_CONFIG.json
# 配置site.paths.native_library=BUILD，execution.finite_check_packing=packed。
python scripts/check_column_repeatability.py --config NEW_CONFIG.json --output-dir NEW_RESUME
```

GDN用family=gated_delta；混合图用case=many-lh/family=mixed；client分别python-native/pytorch。
B4/L16、宽度/区域/k等保持生成器默认；六个完整原配置已内嵌机器记录。
检查只调用实际候选，不借参考轨迹补写恢复。新进程恢复不能代替独立数学公式，反之亦然。
初次产物配置生成误用系统Python3.9的注解错误单列为启动错误，改用项目现行3.12环境后通过。
