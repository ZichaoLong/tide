# 当前NPU native支持修复

用户授权的基座修复已实现，`src/tide/attention.py` SHA256为 `c163d02d95c7e62d6cf9ed83638bfea79cdb354253bcebd7ca104681a954257c`。公开auto和显式npu_fusion使用同一语义包装：单正支持/空行解析VJP，混合行native余切分流；FP32多key增加公共V原点中心化，保留融合内核。普通CPU/CUDA分派及BF16多key算术不变。

单支持归一化输出直接取V，Q/K/归一化正先验梯度为连接的精确零；质量振幅、真实零先验右导数和极弱先验路径保留。公共原点作为固定数值坐标，其数学导数在平移前后相消；V仍收到完整概率导数。范围检查防止平移新增有限数值溢出。没有梯度大小或概率接近1的裁剪。

定向CPU172通过、3原xfail；实际NPU39通过。42个有限NPU状态样本完成，四个旧状态各接续16步并通过首尾8个FP64抽查；全部fresh-process恢复在原atol=1e-5/rtol=1e-4内，最大差5.6904e-7。局部独立公式与同实现CPU FP64整模型诊断的资格范围分开。

最初仅单支持修复的多keyAdam反例保留；原Shannon恢复、CPU私有KV、弱质量随机夹具超差也保留。参数/private-state/checkpoint格式不变；不认证旧学习结论、完整数值范围或10M+ token学习质量。

完整下游记录和源码/日志绑定见[修复审阅](../../tide-learning-lab/docs/pilot-r3-native-auto-repair.md)与[JSON](../../tide-learning-lab/docs/evidence/pilot-r3-native-auto-repair.json)。下游A r08仅推进原表的270个未开始主体及匹配对照，74个旧曝光主体另作迁移审阅，B关闭。运行使用冻结源码，工作树、数学快照和共享环境保持分离。
