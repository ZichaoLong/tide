# 常驻 KV 内容检索完整 Mem 资格

源码 `8b1be80709e501447875ecf3f729afd0a90c5143`；上游 `0770593`。[契约](../resident-weighted-kv-query.md)；[机器记录](core5-resident-kv-query-20261027.json)。

CPU38项为query元数据7、旋转7、窗口6、池化14、projection norm4。
NPU覆盖有限/无限窗口、两次输入/owner/参数版本与五种根组合，共120行；逐条独立公式/autograd比较四返回、全部输入/raw字段/参数bank及None/零。
短期数值下溢后的常值记忆向量与仍连接的写入总量独立检查；空/零长期记忆不读取u/owner。修改原参数/输入/raw后用保存版本反向，容量/非法输入明确拒绝。
启动脚本误写测试路径的首次任务保留failed；未更改测试容差或输入制造通过。

CPU 38 项（4.17 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
