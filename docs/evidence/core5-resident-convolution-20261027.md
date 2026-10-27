# 常驻固定时间短卷积资格

源码 `f89bdfee0fd4ce57803e75bcd5f3924af448289e`；上游 `0770593`。[契约](../resident-weighted-convolution.md)；[机器记录](core5-resident-convolution-20261027.json)。

CPU包含12项卷积和14项raw作用检查。独立dense logical-time signal + grouped conv1d校验全部返回和一阶导数；K=1/2/4、stride=1/3、教材弱槽、空档/余数、超大int64。
NPU覆盖两种K、20行重放、三个共享bank、全部私有cache/当前特征/参数VJP、None/零、动态owner/长度与原来源改写后的独立反向。raw update→新快照卷积→两层VJP保持当前未加权项和连接零质量根。
缺失tap临时乘积可溢出但实际返回梯度有限的边界已修复并单列CPU/NPU回归。容量拒绝、错误窗口不提交通过。
最初检查编译的conv1d重载歧义保留原失败树/日志；未改数值容差。

CPU 26 项（2.04 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
