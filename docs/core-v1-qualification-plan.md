# SettleGraph core-v1 资格计划

> 本文把当前已经闭合的语义子集转成可执行的资格工作单。它不增加模型语义，也不证明仓库已经通过任何资格 cell。
>
> 模型含义以[实验语义、命名与数学符号](experiment-semantics-and-naming.md)为准，公式、比较器和完整目标以[等价性测试契约](equivalence-test-contract.md)为准，实现边界以[SettleGraph 实现与等价性验证计划](settlegraph-implementation-plan.md)为准。三者有冲突时，本计划只能缩小一次声明的范围，不能改写它们的语义。

本文是可执行的验收约束，不声称只凭文本就能重建每个 Tensor byte。本文已给出精确算法的部分必须按该算法实现；人工 golden 数值、非 probe 配置及其他仅给出有限约束的部分，由版本化 generator 在任何被测 executor 运行前一次性物化，通过代码审查后以 canonical bytes、生成决策记录和内容 hash 冻结。后续 executor 与 backend 必须消费这些相同冻结 bytes，不得根据运行结果重选数值、carrier 或子集。因此“字典序最小”只对文中已定义的有限 candidate 表生效，不把未定义的生成器选择暗中当作规范。

## 1. 资格范围和声明边界

### 1.1 core-v1

本文把以下共同范围称为 `core-v1`：

| 坐标 | `core-v1` 的唯一取值或取值集合 |
| --- | --- |
| SettleGraph site | 单 site；测试记录使用固定外部标签 `site.core`，它不是 logical Plan v1 的新字段 |
| logical Plan | schema `1`，canonicalizer `tide-plan-json-v1`，fully expanded 标准 Plan |
| selector context/history | `context.none.v1` 和 `history.none.v1` |
| 参数关系 | 每个 logical parameter key 独立；`shared_parameters=false`，没有参数组 |
| 首状态策略 | 新序列使用当前公式定义的零 Tensor 或空 Attention 窗口；可装载非零的调用前当前状态，但不把它称为 Plan 声明的固定首状态 |
| receiver state | none、EMA、Gated DeltaNet、规范窗口 Attention；每个可变状态只有一个 receiver owner |
| profile/timing | N/content、SD/content、SD/pre、BO/content、BO/pre、BO/post |
| active budget | `k.fixed.v1` 或 `k.input.v1`；不是从模型 Tensor 推导的 adaptive budget |
| dtype binding | 四个核心 dtype roles 全部绑定为 FP64，或全部绑定为 FP32 |
| 局部实现 | eager 标准 Torch 参考公式；没有 mixed precision、compiled、custom kernel 或静默 fallback |
| Base 边界 | 独立 SettleGraph；不把 Qwen、Dense 或 Flat MoE 计入本范围 |

一个通过报告必须写成“`core-v1` 的某个 capability cell 通过”，不能缩写成“完整 SettleGraph 已通过”。[等价性测试契约](equivalence-test-contract.md)第 7 节中的 selector-history、共享只读参数、可学习首状态和多 site 目标仍属于完整目标，见第 13 节的 schema v2 阻塞项。

独立 SettleGraph fixture 固定 $\alpha_{\mathrm{LM}}=0$。LM target mask 只验证输入契约，不能从 hidden 合成语言模型损失；Qwen 专项门才允许非零 LM loss。

非零调用前状态只表示 fixture 从一个已存在序列的已提交状态继续。当前 Plan 没有“固定非零首状态”的策略字段，因此 reset 后回到该非零值、把它作为参数训练或由多个序列共享梯度，都不属于 `core-v1` 声明。

### 1.2 当前执行路径不等于资格完成

逐 Token/token-major eager 路径是调度 reference。当前 region-major eager 路径独立安排 region 顺序，但仍按 region、Token 和 batch row 运行 Python 循环。它不是通用 packed prefill，不能给 `packed` capability cell 提供通过证据。

因此执行器证据分成三类：

1. 逐 Token 路径可接受人工 golden、invariant 和自身 deterministic replay 的 reference 资格；
2. token-major 与 region-major 的 256/64/16 差分只能作为开发回归；
3. 只有真正没有热路径逐 Token、逐 batch row 或逐 node Python 调度的通用 packed executor，对全部适用语料通过后，才能晋级 `packed` cell。

特化 executor 只对其静态支持谓词接受的 Plan 建独立 cell。它不得代替通用 packed executor，也不得只挑运行后已知会通过的样例。

### 1.3 extension-v2

以下能力在 Plan、parameter、trace 或 checkpoint schema 闭合前不能混入 `core-v1`：

- selector-history；
- node、region 或 site 间的只读参数组共享；
- Plan 声明的固定非零或可学习首状态；
- 多 site 参数和状态身份；
- 从 Tensor 推导的 adaptive budget；
- 混合或低精度 accumulation roles；
- 尚未注册完整公式、导数或状态时序的自定义操作。

实现明确拒绝这些能力是合法的 `unsupported` 或 `implemented` 边界，不是 `core-v1` 失败。把它们接受后静默改成无 history、无共享、零首状态或 FP32 eager 则是失败。

## 2. 计数单位和不可变身份

### 2.1 基本对象

- 一个 legal fixture 是一个 logical Plan、一个 concrete dtype binding、一组 CPU 数值输入和一个单调用期望的不可变 bundle。FP64 与 FP32 bundle 属于同一 fixture family，但各自具有 typed Plan hash 和 Tensor artifact hash。
- 一个 scenario 是对一个或多个 bundle 执行的有序动作序列。动作包括调用、reset、release、保存、加载、故障注入和受控并发屏障。scenario 是测试工件，不是新的模型语义。
- 一个 selector event 的稳定键为 `(fixture_id, site_label, sequence_id, token_position, region_id)`。
- 一个 node event 在 selector event 键后追加 `node_id`；一个 edge event再追加 `edge_id`。
- 一个 output event 的稳定键为 `(fixture_id, site_label, sequence_id, token_position, "output")`；它与任何 region/node key 不共享命名空间，并保存按稳定 node ID 排列的实际终端消息 IDs。无执行位置不产生 output event。
- 一个 formula coverage event 在相应 selector、node 或 output event 键后追加 Plan field，例如 `score`、`input_norm`、`update`、`emit` 或 `output_aggregate`；它只区分同一语义事件中不同的公式角色。
- 一个 coverage probe 是 legal fixture 预先指定的一个 reached、候选非空的 selector event，以及其中一个预先指定的 active node。它只用于无歧义地给该 fixture 的 pairwise 行赋值，不改变执行。

site 标签固定为 `site.core`，因为 `core-v1` 只执行一个 site。它进入 trace 和结果排序，但不伪装成 logical Plan v1 或 parameter-schema v1 已支持的跨 site 身份。

### 2.2 计数规则

所有覆盖计数只读取 CPU FP32 token-major reference 的第一次未分块自然路由执行。下列操作不产生额外覆盖次数：

- 用 FP64 再执行同一事件；
- 用 region-major、packed、特化或 NPU 再执行；
- route replay；
- 在多个 chunk 切法中重复读取；
- forward 后又对同一事件 backward；
- 重试失败运行。

topology、shape、layout、mask 和调用前状态来源按 fixture 的 coverage probe 每个各计一次。profile/timing、K 和 route class 按 selector event 键计数；receiver Aggregate、Update、Read、Score、NodeCompute 和 Emit 按实际执行的 node event 键计数；output Aggregate 按 output event 键计数。coverage probe 行的 `A05.output-aggregate` 取同一 `(fixture_id, site_label, sequence_id, token_position)` 下的 output event；该 event 缺失时 probe 失败。一个 event 可以同时贡献不同轴，但不能对同一轴重复贡献。

运行前由 Plan 和期望 trace 产生 `expected-event-coverage.json`；运行后由实际 trace 产生 `observed-event-coverage.json`。两者的键和值必须 exact 相等。预期 probe 没有实际发生时，fixture 失败，不能临时改选另一个 event。

### 2.3 资格身份

一次可晋级证据至少固定：

- 本计划版本和等价性测试契约版本；
- exact commit，以及相关 tracked/untracked 文件均干净的状态；
- generator、canonicalizer、fixture/scenario/result schema 版本；
- corpus manifest、全部 bundle、Plan、参数和输入的 SHA-256；
- executor、公式实现变体、backend、host architecture、精确 accelerator SKU、logical device index 和 dtype；
- 完整 argv、resolved backend、`resolution_reason`、确定性和精度设置；
- comparator 版本、逐稳定路径最坏误差、运行状态和 artifact hash。

本计划不使用“当前文档”或“最新 schema”作版本。corpus 冻结时必须把下表键值逐字写入 manifest，并对三份 authority documents 另存当时 UTF-8 bytes 的 SHA-256：

| identity 键 | 本版字面值 |
| --- | --- |
| `qualification_plan_id` | `tide.core-v1.qualification-plan.v2` + `qualification_plan_sha256` |
| `semantic_authority` | `docs/experiment-semantics-and-naming.md` + `semantic_authority_sha256` |
| `equivalence_contract_id` | `tide.settlegraph.equivalence-contract.document.v1` + `equivalence_contract_sha256` |
| `implementation_plan` | `docs/settlegraph-implementation-plan.md` + `implementation_plan_sha256` |
| `plan_schema_id` | logical schema `1` + canonicalizer `tide-plan-json-v1` |
| `fixture_schema_id` | `tide.settlegraph.fixture.v1` |
| `parameter_schema_id` | `tide.parameter-schema.v1` |
| `failure_schema_id` | `tide.failure.v1` |
| `corpus_schema_id` | `tide.core-v1.qualification-corpus.v2` |
| `axes_schema_id` | `tide.core-v1.axes.v1` |
| `coverage_schema_id` | `tide.core-v1.event-coverage.v1`；record 另含取值为 `expected` 或 `observed` 的 `kind` |
| `result_schema_id` | `tide.core-v1.qualification-result.v1` |
| `run_schema_id` | `tide.core-v1.qualification-run.v1` |
| `negative_schema_id` | `tide.settlegraph.negative.v2` |
| `scenario_schema_id` | `tide.settlegraph.scenario.v1` |

authority documents 当前没有自带的机器可读 schema version，因此上表的 document label 只是本计划的证据坐标，不冒充它们文件内已存在的字段；内容 hash 才固定精确版本。任何会改变解析、计数或通过条件的修订都必须换新相应 schema ID；不能只依赖 hash 变化却继续沿用旧 ID。上表 corpus/axes/coverage/result/run/negative/scenario schemas 未全部实现前，相应 cell 保持 `planned`。

dirty tree 可以产生开发记录，但不能把 capability cell 标为 `verified`。物理卡号、私有工作目录和可见设备映射只进入清洗后的 site-private 运行记录，不进入共享 corpus 或本文。

## 3. 确定性数据来源和冻结工件

### 3.1 唯一随机域

资格生成器使用 ASCII 域 `tide.core-v1.qualification.v2`。对任意字符串标签 $l$ 和非负索引 $i$，定义

$$
U(l,i)
=
\operatorname{uint64}_{\mathrm{big}}
\left(
\operatorname{SHA256}
(\text{domain}\mathbin\Vert 0x00\mathbin\Vert l\mathbin\Vert 0x00\mathbin\Vert\operatorname{decimal}(i))
[0{:}8]
\right).
$$

式中 domain 是所示 ASCII 字面值的 bytes；$l$ 必须是不含 U+0000 的 Unicode scalar sequence，先做 NFC 规范化再编码为 UTF-8；`decimal(i)` 是 ASCII 十进制非负整数，除数值 0 只编码为 `0` 外不带前导零。$\operatorname{uint64}_{\mathrm{big}}$ 把 digest 的前 8 bytes 解释为无符号大端整数。不满足该标签编码的输入在生成前失败，不做替换字符处理。

所有 topology choice、参数、hidden、状态和 cotangent 都从不同标签域取值，不能依赖 Python、Torch、NumPy 或 C++ PRNG 的版本行为。普通浮点源值使用可精确表示的 dyadic 集合

$$
x(l,i)=\frac{(U(l,i)\bmod 2049)-1024}{512}.
$$

需要正数、非零范数或公式值域时，生成器使用本节明确的约束过滤，再取下一个索引；不得在一次运行中随机重采样。每个被跳过的索引和原因写入 generation manifest。

普通的 $1/512$ 网格不能构造 FP32 comparator 的微小 route guard band，因此 route boundary 使用独立的精细 dyadic 域。对按稳定 node ID 升序排列的 candidates $v_1,\ldots,v_C$ 和 $C>K$ 的 probe，除下述 constant 特例外，目标排名取 $v_1,\ldots,v_C$，并物化

$$
a_{v_j}
=
\begin{cases}
(K-j)2^{-4}, & j<K,\\
0, & j=K,\\
-\delta, & j=K+1,\\
-\delta-(j-K-1)2^{-4}, & j>K+1.
\end{cases}
$$

exact-tie 取 $\delta=0$，near-boundary 取 $\delta=2^{-19}$，margin-safe 取 $\delta=2^{-8}$。因此 exact-tie 在边界上由 $v_K<v_{K+1}$ 的稳定 ID 规则唯一决定 Top-K，其他 logits 按 $2^{-4}$ 的可精确表示阶梯远离边界。`score.constant.v1` 不可能实现这个阶梯；它的 exact-tie 专用变体令所有 candidate logits 精确为 0，仍由稳定 node ID 唯一决定 Top-K。constant Score 不进入 near-boundary 或 margin-safe tuple。本计划的 FP32 logit guard band 是

$$
g_K^{32}=4\left[10^{-6}+10^{-5}\max(|a_{(K)}|,|a_{(K+1)}|)\right],
$$

FP64 则把式中的两项阈值替换为 $10^{-10}$ 和 $10^{-8}$。由此 near-boundary 的差 $2^{-19}<g_K^{32}/2$，margin-safe 的差 $2^{-8}$ 大于两种 binding 各自的 $16g_K$。若某个 Score/Read/topology tuple 不能在不共享参数的条件下精确实现这些 probe logits，该 tuple 不是可选 candidate；不得以普通网格过滤后得到的近似值替代。all-active 不构造 K/K+1 边界，其 Score 值继续使用普通数值域。

FP64 直接物化该逻辑源值。FP32 使用 IEEE-754 round-to-nearest-even 独立物化；CPU 与 NPU 必须读取 byte-identical 的 CPU FP32 bundle。参数初始化不能再调用 executor 自己的默认初始化器。

### 3.2 冻结目录的逻辑内容

资格 corpus 发布时至少包含以下相对工件；具体仓库目录可由实现计划确定：

```text
corpus-manifest.json
axes-registry.json
legal-pairs.json
covered-pairs.json
legal/
vjp/
optimizer/
invalid/
scenarios/
golden/
npu-subsets.json
```

`corpus-manifest.json` 保存 256 个 family ID、logical Plan hash、FP64/FP32 typed Plan hash、bundle hash、coverage probe、生成索引和子集成员。它还保存一个 `members` object：键覆盖 corpus 根目录下除 `corpus-manifest.json` 本身外的每个 regular file，值是该文件原始 bytes 的小写 SHA-256 hex。键是规范 POSIX 相对路径：UTF-8 NFC、不得为空，不含 U+0000、反斜杠、绝对前缀、`.` 或 `..` 路径段；symlink、device、socket、FIFO 和 hard-link 的重复 inode 全部拒绝。

顶层 `corpus_root_sha256` 的输入以 ASCII bytes `tide.core-v1.qualification-corpus.v2` 加一个 `0x00` 开头。对所有 member 按路径 UTF-8 bytes 字典序排列，每项依次追加路径 byte 长度的无符号 64-bit 大端编码、路径 bytes，以及 member hex digest 解码后的 32 个 raw bytes；最后对整串做 SHA-256。`corpus-manifest.json` 内保存这个 root digest，但不保存自己的 digest，因而不存在自引用。manifest 按其 schema canonical JSON 物化后的 `corpus_manifest_sha256` 只由 run record 和外层 release index 保存。这两个 digest 共同构成 corpus identity；axes、pairs、goldens、scenarios、negative 和 NPU subset 等非 family 文件也都在 `members` 中，任何成员改变都必须发布新 corpus identity，不能覆盖旧目录。

每个 final run directory 至少保存 `run.json`、`stdout.log`、`summary.json` 和 `artifacts/`。训练和 benchmark 另存原始 `metrics.jsonl`。失败、中断和取消也必须写终态 `summary.json`；不得留下可被误认成仍在运行的空目录。输出目录已存在时拒绝覆盖，resume 只能使用其专用入口。

### 3.3 I00 evidence-infrastructure 前置门

语义比较只有在输入和结果工件可信时才能作为证据。`I00-evidence-infrastructure` 固定为下表 35 个 CPU-safe probes，不计入 256 legal、第 8 节的 96 个 Plan/运行期输入 mutants 与 8 个 artifact mutants、checkpoint negatives 或 runtime negatives。它必须在任何 capability cell 标记 `verified` 前通过；它失败不自动证明数学公式错误，但会使本次证据不可晋级。

| 组（数量） | 固定 probe IDs | 必须结果 |
| --- | --- | --- |
| schema-first（6） | `i00-schema-expected-outcome-list`、`i00-schema-gradient-path-list`、`i00-schema-routing-class-list`、`i00-schema-source-kind-list`、`i00-schema-source-identifier-list`、`i00-schema-parameter-field-list` | 在 raw JSON 中依次把 `expected.outcome`、第一个 `gradient.path_assertions` value、`routing_classification`、`source.kind`、`source.identifier`、第一个 parameter-schema logical-key `field` 改为空 array；全部在 model/state 构造前得到 exact `artifact/artifact.schema`，不泄漏 `TypeError`、`KeyError` 或排序异常 |
| 单 Tensor internal overlap（4） | `i00-overlap-model-expand`、`i00-overlap-model-as-strided`、`i00-overlap-input-expand`、`i00-overlap-input-as-strided` | 前两个把 ID 最小的 eager parameter 替换为 shape 相同的 stride-0 `expand` view 或正 stride 但内部重叠的 `as_strided` view，manifest builder 在输出 manifest 前拒绝；后两个对已认证 fixture 的第一个 parameter Tensor 做同样替换并 reseal，loader 以 `artifact/artifact.schema` 拒绝。四个都不得开始 target copy，不得泄漏 Torch overlap `RuntimeError` |
| fixture 发布（5） | `i00-publish-no-replace`、`i00-publish-two-writer-same`、`i00-publish-two-writer-different`、`i00-publish-post-link-lstat-fault`、`i00-publish-post-link-dir-fsync-fault` | 已存路径的 bytes/inode 不变；两进程在 barrier 后竞争时恰有一个成功，最终文件必须是胜者完整、可认证的 bytes；在 hard-link 成功后向第一次 destination `lstat` 或父目录 `fsync` 注入一次性失败时，writer 只在 destination 仍是自己已发布 inode 时 unlink，再尽力 fsync 父目录；最终无 destination/临时名残留且同路径重试成功。cleanup 的 secondary exception 只追加到 primary failure，不改写 primary 类型/标识 |
| runner 终态（6） | `i00-runner-setup-fault`、`i00-runner-workload-fault`、`i00-runner-final-snapshot-fault`、`i00-runner-metrics-fault`、`i00-runner-keyboard-interrupt`、`i00-runner-hostile-secondary` | 在 run directory 创建后分别对首个 artifact write、workload 入口、第二次 source snapshot、metrics append 做一次性普通失败注入，并在 workload 的第一个 case callback 注入 `KeyboardInterrupt`；五者均非零退出并写与 `failed`/`interrupted`、`setup`/`workload`/`finalize` 一致的 terminal `run.json` 与 `summary.json`。hostile-secondary 令 workload 抛唯一 `PrimaryMarker` 后再令诊断 source snapshot 抛 `SecondaryMarker`，terminal writer 仍成功，对外重抛必须是同一 primary exception object，secondary 只按发生顺序记录 |
| 固定 corpus 完整性（2） | `i00-fixed-corpus-skip`、`i00-fixed-corpus-missing-id` | 把一个预定 case 标记 skip，或从 discovered case IDs 中删掉字典序最大的一项；runner 必须非零终态失败，保存 expected/observed/skipped/missing ID lists，不得以“其余通过”继续任何资格 cell |
| 工件身份与梯度闭包（6） | `i00-digest-bytes-mapping-collision`、`i00-fixture-surrogate-metadata`、`i00-gradient-empty-contract`、`i00-gradient-foreign-key`、`i00-gradient-missing-parameter`、`i00-storage-hole-tamper` | 在不更新 `content_hash` 时把 bytes 换成形似其旧 tag 的普通 mapping，loader 必须以 `artifact/artifact.integrity` 失败；metadata 中的 Unicode surrogate 必须以 `artifact/artifact.schema` 失败，不泄漏 `UnicodeEncodeError`；梯度 contract 的空键集、伪键或漏掉任一实际 parameter 均以 `artifact/artifact.schema` 失败；只改 noncontiguous Tensor backing storage 中不可见 hole byte 而不更新 manifest 时以 `artifact/artifact.integrity` 失败 |
| parameter binding 身份（2） | `i00-binding-executor-spoof`、`i00-binding-state-dict-swap` | 把 eager binding 的 `executor_id` 换成另一个合法 stable ID，或用 `state_dict` post-hook 交换两个同 shape/dtype locator 的 Tensor；manifest 验证必须在 bundle 发布前拒绝，不得只比较 locator 键集、shape 和 dtype |
| failure JSON 唯一解析（1） | `i00-failure-json-duplicate-key` | 在原始 `tide.failure.v1` JSON 的根 object 或嵌套 object 中写入重复键；parser 必须拒绝，不得使用 first-wins 或 last-wins 恢复一个 envelope |
| runner 代码与运行时身份（3） | `i00-runner-foreign-source`、`i00-runner-default-runtime`、`i00-runner-expected-failure` | 用 `PYTHONPATH` 或预加载 module 试图让 runner 执行另一 checkout 的 `tide` 或 corpus test，必须在 workload 前拒绝；runner 必须强制并用真实 Tensor probe 确认 CPU default device 和冻结的 default dtype，并把它们写入 runtime record；任一 `expectedFailure` 或 `unexpectedSuccess` 都必须产生非零失败终态和精确计数 |

所有 fault 都是进程内的一次性测试 hook；probe 结束后立即恢复真实 I/O 并检查目录、inode、bytes 和哈希。若 terminal record 自身遇到持续不可恢复的存储失败，该 run 不可声称具有终态证据；`hostile-secondary` 只注入诊断次级失败，不伪造“持续介质失败下仍能成功写盘”的承诺。

## 4. Axes registry

`axes-registry.json` 必须逐字物化下表。`NA` 表示该 probe 不执行该轴对应的操作；含 `NA` 的 pair 不进入 pairwise 分母。

| 轴 ID | 值 | probe 上的含义 |
| --- | --- | --- |
| `A00.topology` | `singleton`、`single-layer-r2`、`single-layer-r8`、`chain`、`diamond`、`unequal-path`、`multi-entry-terminal`、`mixed-regions`、`forced-backbone`、`small-hb`、`generated-dag` | fixture 的拓扑族 |
| `A01.profile-timing` | `N/content`、`SD/content`、`SD/pre`、`BO/content`、`BO/pre`、`BO/post` | probe region 的传播与选择时序 |
| `A02.state` | `none`、`ema`、`gdn`、`attention-window` | probe node 的 Update/FFN Read family |
| `A03.selector-read` | `content`、`content-rms`、`content-linear`、`content-state-linear`、`content-state-summary-linear` | probe node 的 selector Read |
| `A04.receiver-aggregate` | `mean`、`edge-softmax`、`edge-affine-mean`、`NA` | probe node 非入口且实际收到消息时的 Aggregate |
| `A05.output-aggregate` | `mean`、`node-softmax` | 本 fixture 的图输出 Aggregate |
| `A06.score` | `fixed-by-node`、`constant`、`read-sum`、`linear`、`mlp` | probe event 的 Score |
| `A07.node-compute` | `identity`、`affine-residual`、`double-residual-swiglu` | probe active node 的完整计算 |
| `A08.emit` | `hard`、`hard-st`、`soft-probability` | probe active node 的发送公式 |
| `A09.k-source` | `fixed`、`input` | probe region 的请求来源 |
| `A10.k-class` | `top-1`、`top-2`、`all` | probe event 的实际竞争类别 |
| `A11.shape` | `b1-t1-d2`、`b1-t6-d3`、`b2-t3-d4`、`b3-t5-d7` | 输入 batch、最大物理长度和 hidden 宽度 |
| `A12.layout` | `contiguous`、`noncontiguous-positive-stride` | loader 交给 executor 的 hidden 布局；shape 与值不变 |
| `A13.mask` | `all-execute`、`prompt-no-lm`、`padding-tail`、`routing-subset` | 三个 mask 的预定义合法关系 |
| `A14.state-origin` | `fresh-zero-or-empty`、`serialized-nonzero-current` | probe sequence 的调用前当前状态来源 |
| `A15.route` | `exact-tie`、`margin-safe`、`near-boundary`、`all-active` | CPU FP32 reference 上的 probe selector event 分类 |
| `A16.norm-formula-ids` | `norm/norm`、`norm/test`、`test/norm`、`test/test` | probe node 的 input/FFN normalization 依次使用 `norm.rms.v1` 或 `TEST-RMSNORM-V1` |
| `A17.fixed-score-formula-id` | `score.fixed-by-node.v1`、`TEST-SCORE-CONST-V1`、`NA` | `A06=fixed-by-node` 时的精确 ID；其他 Score 为 `NA` |
| `A18.stateless-ffn-read-type` | `zero`、`state-default`、`NA` | `A02=none` 时两种都是合法的 `read.ffn.zero.v1` 配置 type；有状态 Update 为 `NA`，因其 FFN Read 已由 state family 唯一确定 |

axis value 只是 coverage record 中的稳定标签，不是可由 executor 自行解释的简写。生成器必须按下表把它们展开成等价性测试契约第 2.2 节的精确规范 `type` 和 formula ID：

| 轴 | axis value 到 Plan 的唯一展开 |
| --- | --- |
| `A02.state` | `none` → Update type `none` / `update.none.v1`，FFN Read 由 `A18` 确定；`ema` → Update type `ema` / `state.ema.v1` 与 FFN Read type `state_default` / `read.ffn.ema.v1`；`gdn` → Update type `gdn` / `state.gdn.v1` 与 FFN Read type `state_default` / `read.ffn.gdn.v1`；`attention-window` → Update type `attention_window` / `state.attention-window.v1` 与 FFN Read type `state_default` / `read.ffn.attention-window.v1` |
| `A03.selector-read` | `content` → `read.selector.content.v1`；`content-rms` → `read.selector.content-rms.v1`；`content-linear` 和 `content-state-linear` →各自规范 type 下的 `TEST-READ-PROJ-V1`；`content-state-summary-linear` → `TEST-READ-STATE-RMS-SUMMARY-PROJ-V1` |
| `A04.receiver-aggregate` | `mean` → `agg.mean.v1`；`edge-softmax` → `TEST-AGG-EDGE-SOFTMAX-V1`；`edge-affine-mean` → 规范 type `edge_linear_mean` 及 `TEST-AGG-EDGE-AFFINE-MEAN-V1` |
| `A05.output-aggregate` | `mean` → `agg.mean.v1`；`node-softmax` → `TEST-AGG-TERMINAL-SOFTMAX-V1` |
| `A06.score` | `fixed-by-node` → 规范 type `fixed` 及 `A17` 指定的 ID；`constant` → `score.constant.v1`；`read-sum` → `score.read-sum.v1`；`linear` → `TEST-SCORE-LINEAR-V1`；`mlp` → `TEST-SCORE-MLP-V1` |
| `A07.node-compute` | `identity` → `node.identity.v1`；`affine-residual` → `TEST-NODE-AFFINE-V1`；`double-residual-swiglu` → `TEST-NODE-SWIGLU-V1` |
| `A08.emit` | `hard` → `emit.hard.v1`；`hard-st` → 规范 type `hst` 及 `emit.hst.v1`；`soft-probability` → 规范 type `softp` 及 `emit.softp.v1` |
| `A09.k-source` | `fixed` → `k.fixed.v1`；`input` → `k.input.v1` |
| `A16.norm-formula-ids` | `norm` 分量表示 `norm.rms.v1`，`test` 分量表示 `TEST-RMSNORM-V1`；两分量依次写入 input normalization 和 FFN normalization |
| `A18.stateless-ffn-read-type` | `zero` → type `zero` / `read.ffn.zero.v1`；`state-default` → type `state_default` / `read.ffn.zero.v1`；两者只在 `A02=none` 时合法；其他 state 的该轴值为 `NA` |

chunk 切法、detach、reset、release、row reorder、并发和 checkpoint action 是 scenario 轴，按第 9 节穷举或定额执行，不用从 256 行 pairwise 中抽样。VJP objective、optimizer 类型和 cold/warm optimizer state 是派生子集轴，分别见第 7 节。

### 4.1 合法约束

一行只有同时满足语义文档、Plan schema 和下列 corpus 约束时才进入 legal tuple 集：

1. N 只能使用 content timing 和 `state=none`。pre/post 必须使用 EMA、Gated DeltaNet 或 Attention state。SD 不允许 post。
2. content timing 只配 content、content-rms 或 content-linear。pre/post 只配 content-state-linear 或 content-state-summary-linear。
3. content-state-linear 只用于固定 shape 的 EMA 或 Gated DeltaNet Tensor state；窗口 Attention 使用 content-state-summary-linear。EMA/Gated DeltaNet 也可以使用 summary read。
4. `state=none` 时 Update 为 `update.none.v1`、FFN Read 的 formula ID 为 `read.ffn.zero.v1`，type 由 `A18` 唯一写入为 `zero` 或 `state_default`，state origin 为 fresh empty；它可以出现在 SD/content 或 BO/content 以覆盖合法但 commit 为空操作的配置。
5. forced-active singleton 固定 `K=1`、`k-source=fixed`、`k-class=all` 和 `route=all-active`。其他 `k-source=input` event 必须在 selector 前提供 int64 值。
6. `top-1` 要求候选数至少 2 且实际 $K=1$；`top-2` 要求候选数至少 3 且实际 $K=2$；`all` 要求 $K\ge C$。exact-tie、margin-safe 和 near-boundary 都要求 $C>K$。
7. exact-tie 由逻辑源 logits 精确相等构造。margin-safe 在 FP64 和 FP32 两个 binding 中都满足 $\Delta_K$ 大于各自 guard band 的 16 倍。near-boundary 在 CPU FP32 中满足 $0<\Delta_K\le g_K/2$，并要求自然 route 仍与预期 exact 相同。具体 logit 数值使用第 3.1 节的精细 dyadic 域。
8. VJP、gradcheck 和 optimizer 子集不使用 near-boundary。凡可微路径经过不带 epsilon 的 RMS read，fixture 必须令相应向量范数至少为 $2^{-8}$；L2 normalization 输入范数与 `norm_eps` 的距离至少为 $2^{-8}$。零范数 RMS 只进入 forward cell。
9. edge-softmax 和 edge-affine mean 的有效计数只来自非入口 probe node，且至少有两条实际 `DATA` 父消息；否则该轴为 `NA`。node-softmax 的有效计数要求至少两个 active terminal messages。
10. noncontiguous layout 必须保持正 stride、无内部 overlap，并在 manifest 保存 shape、stride 和 storage offset。负 stride、overlap 和稀疏 Tensor 不是 `core-v1` 输入。
11. serialized nonzero current state 必须具有合法 owner、shape、dtype 和连续 next position；Attention 保存按位置排序的非空有效窗口。它不改变 reset 首值。
12. 每个 Plan 的所有局部公式必须来自等价性测试契约第 2.1 和 2.2 节的 reference registry。`TEST-NORM-IDENTITY-V1` 尚未注册，不进入 `core-v1`。
13. `A16` 的斜线前后分别是 input normalization 和 FFN normalization 的 formula ID；规范化和 parameter manifest 保留原 ID。`A17` 只在 fixed-by-node Score 上有值，两个同义 Score IDs 不互相重写。`A18` 则保留 stateless FFN Read 的两个合法规范 type，不因 formula ID 相同而合并。
14. `prompt-no-lm` 和 `routing-subset` 要求 $T\ge2$；`padding-tail` 要求 $B\ge2$ 且 $T\ge2$，从而相应 mask 值在实际 Tensor 上可观察，不把退化为同一 Tensor 的标签计作不同覆盖。

四种 mask 值使用固定生成规则。令物理列为 $t=0,\ldots,T-1$：`all-execute` 令三个 masks 全 true；`prompt-no-lm` 令 execution/routing 全 true，LM mask 仅在 $t\ge\lceil T/2\rceil$ 为 true；`padding-tail` 对 row $b$ 取有效长度 $L_b=\max(1,T-b)$，execution/routing 在 $t<L_b$ 为 true，LM mask 还要求 $t>0$；`routing-subset` 令 execution 全 true、LM mask 要求 $t>0$、routing mask 要求 $(b+t)\bmod2=0$。所有 true execution positions 的全局位置连续；false 位置使用不参与校验的 int64 哨兵。

noncontiguous hidden 从 shape 为 $[B,T,2d]$ 的无 overlap contiguous backing Tensor 取 `[..., ::2]`，然后按 logical source values 填充该 view；FP64/FP32 bundle 都保存其 stride 和 storage offset。fresh state 的 next position 为 0；serialized current state 的 next position 固定为 3，本次 true positions 从 3 连续开始，其 Tensor 或 Attention 有效窗口由 `state:<fixture_id>` 域生成且至少一个元素非零。

`legal-pairs.json` 由完整合法 tuple 集产生：对任意两个不同轴，若至少存在一个不含 `NA` 的合法完整 tuple 同时取这两个值，该值对就是必须覆盖的 legal pair。不能因为最终 256 行没有选到某一对就把它从分母删除。

数学同义的公式 ID 仍由 `A16` 和 `A17` 作为 exact Plan configuration 分别覆盖；同 formula ID 下的合法 type 别名由 `A18` 分别覆盖。每个具体 `(field,type,formula_id)` 单独进入 expected/observed event coverage 和 parameter manifest，不能在 canonicalization 或报告中改写成另一个配置。

## 5. 256 个 legal fixture 的确定性构造

### 5.1 固定槽位

256 个 family 使用 `ql-0000` 至 `ql-0255`。槽位先固定 topology，再分配其余轴：

| 槽位 | topology | 数量 | 固定模板 |
| --- | --- | ---: | --- |
| 0000–0015 | singleton | 16 | 一个入口兼终端 receiver，一个 forced-active singleton region |
| 0016–0031 | single-layer-r2 | 16 | 两个 receivers 同为入口和终端，单 region |
| 0032–0047 | single-layer-r8 | 16 | 八个 receivers 同为入口和终端，单 region |
| 0048–0063 | chain | 16 | 至少三层、每层一个 region |
| 0064–0079 | diamond | 16 | 一处分叉、一处汇合，汇合父边按稳定 edge ID 排列 |
| 0080–0095 | unequal-path | 16 | 同一汇合点前含长短路径和合法 skip edge |
| 0096–0111 | multi-entry-terminal | 16 | 至少两个入口和两个终端 |
| 0112–0127 | mixed-regions | 16 | 同一拓扑层含 singleton 与竞争 region |
| 0128–0143 | forced-backbone | 16 | forced backbone 保证终端，旁路分支可全部关闭 |
| 0144–0159 | small-hb | 16 | fully expanded Lines、barrier 和 edge-source labels |
| 0160–0255 | generated-dag | 96 | 有界分层 DAG |

每个手工 topology 的 16 个变体必须在至少一个实际 reached 的语义项上不同，不能只改 `plan_id`、Builder metadata 或未执行常量。生成器用槽位号派生 normalization epsilon、合法状态常量、边/region 变体和公式参数。schema 1 的 logical Plan canonical bytes 本就不含仅作 provenance 的 `plan_id` 和 `builder`；本文不再定义第二种含义重叠的“semantic fingerprint”。唯一性定义为 256 个 logical Plan canonical SHA-256 两两不同，且每对手工变体的差异都在预期 trace 的 reached 字段中有对应事件。

为判定 generated DAG 是否重复，单独定义 `tide.core-v1.topology-projection.v1`。它是一个键集 exact 的 JSON object，只含 `schema_id`、`topology_kind`、`entry_node_ids`、`terminal_node_ids`、`nodes`、`edges` 和 `regions`：`nodes` 每项只含 `node_id`、`region_id` 和 `forced_active`；`edges` 每项只含 `edge_id`、`source`、`target` 和 `label`；`regions` 每项只含 `region_id`、`node_ids`、`control_dependencies`、`line` 和 `phase`。每一级 object keys 都按 Unicode code point 升序，三个 entity arrays 和所有 ID arrays 按语义文档的稳定 ID 顺序；字符串必须是 NFC Unicode scalar sequence。JSON 的 string escaping 和 scalar 字面值完全复用 `tide-plan-json-v1` renderer，使用 UTF-8、原样 Unicode、小写 `true`/`false`/`null`、不加空白，不允许额外键。topology fingerprint 是对这串 bytes 的小写 SHA-256 hex。shape、dtype、公式、参数、profile/timing、K 和 Builder provenance 不进入该 projection；影响 barrier 的 control dependency/line/phase 和影响消息语义的 edge label 进入。

### 5.2 generated DAG

对 generated 槽位的绝对整数索引 $i\in\{160,\ldots,255\}$，首次尝试的标签 $p$ 为 `dag:<decimal(i)>:retry:0`。对每次尝试把 draw cursor $q$ 重置为 0，并定义 `draw(m) = U(p,q) mod m`，每次 draw 后 $q$ 恰好加 1。不读取实现语言 PRNG 或一次性 SHA bit buffer。

1. 层数为 $L=3+\operatorname{draw}(3)$。对 $\ell=0,\ldots,L-1$ 依次令 node 数 $n_\ell=2+\operatorname{draw}(3)$，层内索引 $j=0,\ldots,n_\ell-1$，node ID 为 `node.l<ell:02>.n<j:02>`。
2. 每层按 node ID 从小到大扫描。设剩余 node 数为 $r$，下一 region 宽度为 $1+\operatorname{draw}(\min(3,r))$，直到该层分完；region ID 为 `region.l<ell:02>.r<j:02>`。因此 region 只包含同层连续 nodes，宽度为 1 至 3。
3. 对每个非终层 node $j$ 增加到下一层 node $j\bmod n_{\ell+1}$ 的边；再对每个非首层 node $j$ 增加从前一层 node $j\bmod n_{\ell-1}$ 的边。边集按 endpoint pair 去重。
4. 对每对相邻层，按 `(source_id,target_id)` 枚举尚不存在的 pair；每个 pair 恰好消费一次 `draw(16)`，结果小于 3 时增加该边。
5. 对每对相隔两层的层，先消费一次 `draw(2)`；值为 1 时，再消费一次 `draw(n_left*n_right)` 按 endpoint pair 字典序选唯一 shortcut。
6. 入口集恰为第 0 层全部 nodes，终端集恰为第 $L-1$ 层全部 nodes。最后按 `(source_id,target_id)` 排序并分配 `edge.<j:04>` ID；这些 generated edges 的 label 固定为 `data`，node 的 `forced_active` 固定为 false，region 的 `control_dependencies` 为空且 `line`/`phase` 为 null。

第 3 步同时保证每个非终层 node 有出边、每个非首层 node 有入边；由层号严格增加可知图是 DAG，且每个 node 都在入口到终端的路径上。若与较早 generated case 的第 5.1 节 topology fingerprint 重复，则令标签中的 retry 值 $n=1,2,\ldots$ 单调递增，取第一个新 fingerprint；尝试标签、draw 次数和 retry 原因全部记录在 manifest。Plan validation 失败时生成过程失败，不得跳过该槽位换成一个更容易的 topology。

### 5.3 轴赋值和 pairwise

每个槽位根据 topology、probe 位置和第 4.1 节形成有限的合法 axis tuples，按 axes registry 中的值顺序做字典序排列。非 probe nodes/regions 的配置由槽位、tuple rank 和稳定 ID 的 SHA 域确定，并至少让本 case 所声明的 operator coverage events 实际 reached。

选择向量 $R=(r_0,\ldots,r_{255})$ 中，$r_i$ 是槽位 $i$ 的 tuple rank。规范 corpus 取满足下列约束的字典序最小 $R$：

- 所有 `legal-pairs.json` 中的 pair 至少被一个 coverage probe 行覆盖；
- 六种 profile/timing、三种 receiver Aggregate、两种 output Aggregate、三种 Emit、四种 state、两种 K source、三种 K class、五种 selector Read、五种 Score、三种 NodeCompute、四种 shape、两种 layout、四种 mask、两种 state origin、四种 normalization ID pairs、两个非 `NA` fixed Score IDs，以及两个非 `NA` stateless FFN Read types，各至少计 16 次；
- exact-tie 至少 16 个 selector events，margin-safe 至少 64 个，near-boundary 至少 16 个，all-active 至少 16 个；
- 每个 `core-v1` 已注册的 `(field,type,formula_id)` 至少有 16 个实际执行的对应 formula-coverage events；事件底座按第 2.2 节分别是 selector、node 或 output event，因此 output Aggregate 不得用 node event 代计；需要两条消息或两个终端的公式按第 4.1 节的有效事件计数；
- 第 8.3 节 52 个 mutation 的每组 carrier 前置都至少有两个 legal fixtures 满足第 8.2 节的不同-topology/不同-logical-hash 选择规则；对 `owner-alias-object`，carrier 中两个独立 storage 的 state Tensor 预先物化为 shape/dtype/value exact 相同，不在 mutation 时临时改值；
- 第 7 节的 64 VJP 和 16 optimizer 子集存在；第 11 节的嵌套 NPU 子集也存在。

实现可用 SAT、ILP 或回溯求这个有限约束问题，但必须通过逐槽位固定最小可行 rank 得到同一个字典序最小解；solver 名称不成为语料身份。若约束无解，生成器非零失败，并修改计划/生成器版本后重新评审；不得在运行时放松计数或增加第 257 个样例。

`covered-pairs.json` 保存每个 legal pair 的至少一个最小 fixture ID 和 probe event key，另存完整 multiplicity。通过条件是 uncovered list 为空，不能只报告覆盖百分比。

formula ID 由 axis tuple 在 logical Plan canonicalization 前确定。若某个 ID 与 Plan 约束冲突，该 tuple 非法；不得退回到只按语义 operation family 计数。

### 5.4 单个 legal bundle

每个 family 物化 FP64 和 FP32 两份 bundle，至少包含：

- canonical logical/typed Plan bytes 与 hash；
- Plan 派生的 parameter-schema manifest、executor-independent logical keys 和 eager locator binding；
- 由第 3.1 节产生的参数、hidden、调用前状态和运行期 K；
- `sequence_id`、全局位置、三个 masks、layout 元数据和 coverage probe；
- 期望 candidates、K、route IDs、route class、event keys 和 invariant 数量；
- comparator policy 和所有 artifact hashes。

每个 bundle 的 $(B,T,d)$ 由 shape 轴给定。padding 槽位的位置放一个明显越界但被忽略的 int64 哨兵；至少 16 个 fixture 还在候选为空的 downstream event 放越界 `requested_k` 哨兵，并由 trace 证明该位置未读取。

## 6. 人工 golden

资格库长期保存 12 个合法人工 golden 和 2 个故障 golden：

| ID | topology/场景 | 独立展开的重点 |
| --- | --- | --- |
| `golden-00` | singleton | 实际 Read/Score、单元素 logit、精确 probability 1、forced active、identity 输出 |
| `golden-01` | R2 Top-1 exact tie | node ID 平票、hard Emit、终端均值 |
| `golden-02` | R8 Top-2 margin-safe | Top-K 顺序、Hard-ST 前向与 probability 梯度 |
| `golden-03` | R8 all-active + input K | all-active、soft-probability Emit、K 读取时点；一个 stateless active node 使用 type `state_default` / `read.ffn.zero.v1` 并在 NodeCompute 中实际读取 |
| `golden-04` | chain | region 顺序、EMA carry、下一位置 |
| `golden-05` | diamond | fan-out、`CLOSED`、edge order、edge-softmax fan-in |
| `golden-06` | unequal path | 短路径缓存、shortcut 和晚汇合 |
| `golden-07` | multi-entry/terminal | 边界广播、terminal node-softmax |
| `golden-08` | mixed regions | 独立 ready regions、singleton 与竞争 event |
| `golden-09` | forced backbone | 可选分支全关仍有终端消息 |
| `golden-10` | small HB | Lines、barrier、tree/local/shortcut/mirror labels |
| `golden-11` | masks/empty candidate | prompt、padding、routing subset、越界 K 哨兵未读取 |
| `fault-golden-00` | late local operation | 早期 staged write、晚期失败、整调用回滚 |
| `fault-golden-01` | injected empty terminal | 防线失败、无公开 output/stat/state 发布 |

`golden-formula-coverage.json` 以 `(Plan field, 规范 type, formula_id)` 为键，不只按相似的 operation family 合并。每个键必须指向下表规定的至少一个 golden ID、实际 event key、可读推导段落和 expected artifact path；该 event 必须 reached 并真正执行该公式。不能用一个未被 NodeCompute 读取的 FFN normalization、只有一条消息的 learned Aggregate，或未 reached node 填补 coverage。

| 公式键 | 必须提供独立 oracle 的 golden |
| --- | --- |
| input/FFN normalization 的 `norm.rms.v1` 和 `TEST-RMSNORM-V1` | `golden-02` 依次使用 input `norm.rms.v1` / FFN `TEST-RMSNORM-V1`，`golden-08` 使用 input `TEST-RMSNORM-V1` / FFN `norm.rms.v1`；两个 FFN 位置都必须被 SwiGLU 实际读取 |
| receiver Aggregate `agg.mean.v1`、`TEST-AGG-EDGE-SOFTMAX-V1`、`TEST-AGG-EDGE-AFFINE-MEAN-V1` | 依次 `golden-04`、`golden-05`、`golden-06`；后两者的实际父消息至少两条 |
| output Aggregate `agg.mean.v1`、`TEST-AGG-TERMINAL-SOFTMAX-V1` | 依次 `golden-00` 和 `golden-07`；后者至少两个 active terminal messages |
| Update `none` / `update.none.v1`；FFN Read `zero` / `read.ffn.zero.v1` 与 `state_default` / `read.ffn.zero.v1`；`ema` / `state.ema.v1` 与 `state_default` / `read.ffn.ema.v1`；`gdn` / `state.gdn.v1` 与 `state_default` / `read.ffn.gdn.v1`；`attention_window` / `state.attention-window.v1` 与 `state_default` / `read.ffn.attention-window.v1` | stateless Update 和 type `zero` Read 用 `golden-02`，stateless type `state_default` Read 用 `golden-03`，其余三个有状态 pair 依次用 `golden-04`、`golden-08`、`golden-11`；两个 stateless Read 都被 NodeCompute 实际读取，有状态三项同时手算 proposal、commit 和实际 FFN readout |
| selector Read `read.selector.content.v1`、`read.selector.content-rms.v1`、content `TEST-READ-PROJ-V1`、state `TEST-READ-PROJ-V1`、`TEST-READ-STATE-RMS-SUMMARY-PROJ-V1` | 依次 `golden-00`、`golden-01`、`golden-02`、`golden-04`、`golden-11` |
| Score `score.fixed-by-node.v1`、`TEST-SCORE-CONST-V1`、`score.constant.v1`、`score.read-sum.v1`、`TEST-SCORE-LINEAR-V1`、`TEST-SCORE-MLP-V1` | 依次 `golden-00`、`golden-01`、`golden-03`、`golden-02`、`golden-04`、`golden-08` |
| NodeCompute `node.identity.v1`、`TEST-NODE-AFFINE-V1`、`TEST-NODE-SWIGLU-V1` | 依次 `golden-00`、`golden-05`、`golden-02` |
| Emit `emit.hard.v1`、`emit.hst.v1`、`emit.softp.v1` | 依次 `golden-01`、`golden-02`、`golden-03`；Hard-ST 的 surrogate derivative 另按第 7.1 节解析验证 |
| `context.none.v1` / `history.none.v1` 与 `k.fixed.v1` / `k.input.v1` | none 语义用 `golden-00`；两种 K 依次用 `golden-01` 和 `golden-03` |

这些 fixture 使用 $d\in\{2,3\}$、长度至多 4 的 dyadic 数值。期望生成器只能使用语言内标量四则运算、明写循环和独立的高精度 `exp`/`sqrt`；不得导入被测 Aggregate、Update、Read、Score、Top-K、NodeCompute、Emit、state commit、balance-loss 或 executor helper。每个公式同时保存可读推导和完整 expected trace，不能只保存最终 output。

人工期望按等价性测试契约第 3 节保存 absent、edge status、父消息、proposal、readout、logit/probability、K、Observe/active、NodeCompute、Emit、staged state、终端聚合、最终状态和充分统计。离散字段 exact；浮点值按第 10.1 节比较。

这 12 个合法 golden 是独立 oracle 工件，不减少第 5 节的 256 个 legal family 数量；它们可以复用某个 topology 模板，但使用独立 fixture ID 和内容 hash。

## 7. 64 VJP 与 16 optimizer fixture

### 7.1 VJP 子集

从 256 个 legal family 中预先选择 64 个不同 fixture ID，按 legal fixture ID 排序后映射为 `vjp-000` 至 `vjp-063`。它们都排除 near-boundary；凡竞争路由可微输入或参数决定，必须 margin-safe。exact-tie 只允许 Score 对本 fixture 所有进入方向差分的量都结构性独立，且在规定差分邻域内 route 不变；all-active 没有 K/K+1 边界。VJP ordinal 对 8 取模后索引下表，因此每个 objective class 恰有 8 个：

| objective class | 固定标量目标和必须检查的路径 |
| --- | --- |
| `output-hard` | output cotangent；Hard Emit 不增加 probability 路径 |
| `output-hst` | output cotangent；验证 Hard-ST 对 active probability 的局部导数 |
| `output-softp` | output cotangent；验证前向强度和 selector 路径 |
| `final-state` | 可微最终 receiver state cotangent；验证 Update/FFN Read |
| `balance` | `BAL-AVAIL-SOFT`；只有 $P_v$ 返回梯度，其余充分统计 stop-gradient |
| `bo-post-proposal` | 只由 post selector logit/probability 构造；proposal 到 Update 参数可导 |
| `pre-separation` | pre selector 隔离目标不经本 Token proposal返回 Update，active compute 路径仍按公式连通 |
| `chunk-edge` | 同一 forward state carry 的 detach/no-detach 对；只改变跨 chunk 反向边 |

cotangent 用 `U("cotangent:<fixture_id>:<objective>", i)` 产生并保存在 bundle，不使用“对所有输出求和”的隐含默认。目标的非零系数固定为 dyadic 值 $1,1/8,1/16$，完整表达式逐 fixture 保存。

一个 family 只有在它满足对应 ordinal objective 的结构前置时才是该位置的 candidate：`output-hard`、`output-hst` 和 `output-softp` 必须分别实际执行对应 Emit；`final-state` 必须含可微的非空最终 Tensor state；`balance` 必须至少有一个 routing-stat mask 选中、候选非空且概率路径可微的竞争事件；`bo-post-proposal` 必须是含参数化 Update 的 BO/post；`pre-separation` 必须是含参数化 Update 的 SD/pre 或 BO/pre；`chunk-edge` 必须 stateful 且有至少两个连续执行位置。每个目标还必须至少有一个预先声明的 connected key 产生非零梯度；不能用全零 cotangent 或退化参数让路径断言真空通过。

子集选择是字典序最小的有序 64 元组：第 $j$ 个元素的 legal fixture ID 严格递增，其 objective 由 $j\bmod8$ 唯一确定，且该 fixture 必须通过上述位置前置。选择约束还覆盖所有可微公式参数角色、六种 profile/timing、四种 state、三种 receiver Aggregate、两种 output Aggregate、三种 Emit、两种 K source、三种 NodeCompute、五种 Score、输入 hidden 和调用前可微当前状态。fixture v1 的 `required_keys` 必须精确等于 `inputs.hidden`、parameter manifest 中实际存在的全部 logical parameter keys，以及实际存在的可学习初态 keys；`core-v1` 的最后一类为空。每个实际 key 只标注为 `connected` 或 `disconnected`：connected 的零梯度必须是同 shape 零 Tensor，不能是 `None`，disconnected 必须返回 `None`。结构上不存在的参数由 parameter manifest 的 exact key set 证明，不向 `required_keys` 注入没有 Tensor 的幽灵路径。若未来 fixture 需要把特定 structurally absent 路径作为一等 VJP 断言，必须先定义有限、稳定的跨 fixture 路径全集并升级 fixture schema，不能在 v1 中接受任意字符串。

另外从这 64 个 cases 中冻结字典序最小的 32 个 CPU FP64 方向有限差分 cases，不再使用“前 32 个”这个与 objective ordinal 冲突的规则。FD candidate 必须对规定标量 forward 目标有真实的局部导数、全部连续公式远离非光滑点，且 $x\pm hd$ 两侧的 candidates、Top-K IDs 和 route 与中心 exact 相同。32 元集合覆盖所有适用的非退化 objective classes 和可微公式参数角色；若无解则 corpus generation 失败。

`output-hst` 不是 FD candidate：Hard-ST 的 forward 恒等于 $g$，但它声明的 surrogate VJP 本来就不是该 forward 函数的数值导数。它使用不调用 executor Emit helper 的解析局部 oracle。对该 active node 的 Emit 输出 $\widehat g$ 单独注入一个由 bundle 物化的局部 cotangent $\bar{\widehat g}$；此 probe 把 Emit 的三个局部输入 $h$、$g$ 和 $p$ 当作相互独立的 leaves，不经过后续 edge、terminal Aggregate 或整图 output cotangent。它要求 forward $\widehat g=g$，且直接局部 VJP 为 $\bar g=\bar{\widehat g}$、$\bar h=0$以及

$$
\bar p
=
\zeta^{\mathrm{ST}}\langle \bar{\widehat g},g-h\rangle.
$$

整图 `output-hst` objective 仍使用其冻结的最终 output cotangent 检查端到端 VJP，但不把它直接代入上述局部等式；局部 probe 与整图 probe 使用不同 artifact 路径和名称。

`chunk-edge` 的 no-detach branch 是普通 forward 函数，可进入这 32 个 FD；detach branch 不与相同 forward 的有限差分比较，而是要求前一 chunk 的指定 key 在 no-detach 中 `connected` 且产生非零梯度，在 detach 中 `disconnected` 且返回 `None`，两分支的 forward state/output 仍相等。

对一个 FD case，把所有 `connected` 的 hidden、调用前 Tensor state 和 logical parameters 按稳定路径展平串接为 $x$。`disconnected` key 有 Tensor 但不进入主方向 $d$，其 autograd `None` 不被伪造成 connected 零 Tensor。每个 disconnected key 另做一个只在该 key 上非零的独立 dyadic 中心差分 probe，要求两侧标量目标在 `TFD64` 内无变化；这些 key-local probes 写入同一 FD case artifact，不额外计入 32 个 case 数。离散状态元数据、positions 和 optimizer state 不进入 $x$。主方向原值用独立 dyadic 域产生，跳过全零向量，再缩放为 $\lVert d\rVert_\infty=1$。定义

$$
d_{\mathrm{AD}}=\langle\nabla_xJ,d\rangle,
\qquad
d_{\mathrm{FD}}=\frac{J(x+hd)-J(x-hd)}{2h},
\qquad
h=2^{-20}\max(1,\lVert x\rVert_\infty).
$$

每个 disconnected key-local probe 使用同一步长规则，只把上式的 $x,d$ 替换为该 key 的展平 Tensor 和它的独立归一化方向。

中心差分与 autograd 方向导数要求

$$
|d_{\mathrm{AD}}-d_{\mathrm{FD}}|
\le 10^{-6}+10^{-4}|d_{\mathrm{FD}}|.
$$

fixture 在 $h$ 邻域内必须保持同一路由；若不能证明，就不能进入这 32 个样例。解析导数或 `gradcheck` 可以作为额外证据，不能替代固定的 32 个方向差分。

### 7.2 optimizer 子集

从 64 个 VJP fixtures 中预先选择满足约束的字典序最小 16 元集合，按 legal fixture ID 排序后映射为 `opt-00` 至 `opt-15`。选择约束覆盖四种 state、三种 receiver Aggregate、两种 output Aggregate、三种 Emit、六种 profile/timing、两种 K source 和所有实际 trainable parameter roles；无解时 corpus generation 失败，不能换成运行后已知通过的集合。

每个 optimizer case 沿用它在 VJP bundle 中已冻结的唯一标量目标、cotangents 和路径断言，不在 optimizer runner 中改成隐含的 output sum 或重新抽样 loss。一个 fixture 只有在该目标对至少一个 trainable logical parameter key 产生有限非零梯度时才能进入 16 个子集。

- `opt-00` 至 `opt-07` 使用 Adam：`lr=0.003`、`betas=(0.8,0.95)`、`eps=1e-8`、`weight_decay=0`、`amsgrad=false`。
- `opt-08` 至 `opt-15` 使用 AdamW：`lr=0.002`、`betas=(0.9,0.98)`、`eps=1e-8`、`weight_decay=0.01`、`amsgrad=false`。
- 每种 optimizer 的偶数 local ordinal 从空 optimizer state 开始；奇数 local ordinal 装载冻结的、已经完成两次 deterministic priming step 的非零 state。

参数组及组内 logical key 均按稳定字符串排序。priming gradients 由独立 bundle直接携带，不由任一被测 executor 临时生成。每个 case 比较 step 前所有梯度、step 后全部参数、step counter、moment Tensor、参数组超参数和 key 顺序。`core-v1` optimizer cell 不包含共享参数或可学习首状态；这些必须进入 extension-v2 optimizer cell。

## 8. 104 个 negative mutants：96 个 Plan/运行期输入 + 8 个 artifact

### 8.1 负向工件

正向 bundle loader 不能先要求非法 Plan 通过 validation。资格 runner 需要一个独立、版本化的 `tide.settlegraph.negative.v2` container，保存 valid base bundle hash、单一 mutation ID、原始 mutated bytes/record、期望 `tide.failure.v1` envelope 和适用入口。这个测试工件 schema 不改变 logical Plan schema。

actual envelope 必须由 validator、loader 或已知执行阶段产生。比较函数不得把 fixture 的 expected code 作为 actual code 传回，也不得解析异常文本。对一个 mutation，actual phase 和该 phase 的完整排序 code 集必须与 expected exact 相等。

### 8.2 确定性构造

下列 52 个 mutation IDs 各应用到两个不同 legal carriers，恰得 104 个 negative artifacts。每项的第一个 carrier 是满足前置条件的最小 fixture ID；第二个是满足前置条件且 topology fingerprint 不同的最小 ID。若不存在不同 topology，才取下一个 logical Plan hash 与第一 carrier 不同的 ID；仍不足两个则生成失败。前者物化为 FP64 mutant，后者物化为 FP32 mutant。前 4 种 artifact/schema mutation 的 8 个实例用于认证工件边界，独立于[等价性测试契约](equivalence-test-contract.md)第 7.2 节的数量门槛；其余 48 种恰好产生该门槛要求的 96 个非法 Plan 或运行期输入 mutants。动态 mutation 还必须选择预先指定的唯一 reached event，ID 为 `invalid-<mutation-id>-0` 或 `invalid-<mutation-id>-1`。

| 组 | mutation IDs；每项都是一次有名变换 | 期望 phase/code |
| --- | --- | --- |
| artifact/schema | `bundle-content-hash`、`canonical-bytes-hash`、`bundle-root-version`、`tensor-manifest-shape` | 前两项 `artifact/artifact.integrity`；后两项 `artifact/artifact.schema` |
| Plan schema/topology | `plan-root-version`、`illegal-stable-id`、`duplicate-node-id`、`duplicate-edge-id`、`cycle`、`intra-region-edge`、`unknown-edge-endpoint`、`wrong-entry-set`、`wrong-terminal-set`、`foreign-state-owner` | 前两项 `plan/plan.schema`；其余 `plan/plan.topology` |
| formula | `formula-unknown-key`、`formula-missing-required`、`formula-id-type-mismatch`、`bias-false`、`derived-output-shape`、`derived-state-shape`、`n-pre`、`sd-post`、`fixed-k-zero`、`values-by-node-keyset` | 全部 `plan/plan.formula` |
| binding/input | `binding-missing-role`、`binding-symbolic-dtype`、`hidden-shape`、`hidden-dtype`、`duplicate-sequence-id`、`reset-id-duplicate`、`requested-k-region-keyset`、`requested-k-shape`、`execution-mask-shape`、`execution-mask-dtype`、`token-position-shape`、`token-position-dtype`、`lm-mask-outside-execution`、`routing-mask-outside-execution`、`position-replay`、`position-skip` | 前两项 `binding/binding.invalid`；hidden、sequence、reset、K container 和 token-position shape/dtype 为 `input/input.schema`；execution-mask shape/dtype 与两个 mask-subset 变换为 `input/input.mask`；最后两项 `input/input.position` |
| state/event/execution | `state-owner-key`、`state-shape`、`owner-alias-object`、`owner-alias-view`、`requested-k-zero-late`、`requested-k-above-max-late`、`local-update-failure-late`、`local-node-failure-late`、`empty-terminal-first`、`empty-terminal-late`、`attention-window-order`、`attention-window-length` | 前两项和最后两项 `state/state.schema`；两种 alias 为 `state/state.owner_alias`；两种 K 为 `event/input.requested_k`；两种 local failure 为 `execution/execution.local_operation`；两种 empty terminal 为 `execution/execution.empty_terminal` |

`binding-missing-role` 从 concrete `binding.dtype_roles` object 中只删除 `readout` role。`binding-symbolic-dtype` 保留四个 role，但只把 `binding.dtype_roles.readout` 的 concrete 值改为字面 `runtime`；这个值是 logical Plan 可用的符号声明，却不是 concrete binding 允许的 dtype，因此由现有 binding validator 唯一产生 `binding/binding.invalid`。负向 container 保存这个原始 mutated binding record 并重算外层工件 hash，但不伪造一个已通过 validation 的 typed Plan hash；重算外层认证字段不计为第二个语义 mutation。

上表五组分别有 4、10、10、16、12 项，合计 52；每项两个 carriers，因此总数 exact 为 104。其中非 artifact 的后四组共 48 种、96 个实例，恰好满足上位契约的非法 Plan/运行期输入门槛；4 种、8 个 artifact faults 另计。`learnable_decay=true` 与 `shared_parameters=true` 的 v1 拒绝可作为额外开发 mutants，并在 schema v2 引入相应能力时改由新 schema 的正负例覆盖，但它们不计入这 104 个固定实例。

其中前四组 40 种 mutation，加上最后一组的 `state-owner-key`、`state-shape`、两种 owner alias 和两种 Attention window mutation，共 46 种/92 个 mutants 应在静态或 loader 入口失败。两种 late K、两种 local failure 和两种 empty terminal 共 6 种/12 个 mutants 必须先通过 loader，再在 `C11` 运行到预定 event/execution 入口时失败；若它们被更早 gate 意外拒绝，case 也失败。

`requested-k-zero-late`、`requested-k-above-max-late`、`local-update-failure-late`、`local-node-failure-late` 和 `empty-terminal-late` 的每个 case 在失败点之前至少有两个成功执行 Token 和一次 staged state write。`empty-terminal-first` 则专门验证首个执行位置的防线和零公开发布；中途事务回滚由 `empty-terminal-late` 覆盖。失败后比较所有公开 receiver states、next positions、RNG、可见 artifact 路径集合和 checkpoint 可见内容 hash 与调用前 exact 相同；不得返回部分 output 或部分充分统计。原始 mutant 和尽可能收缩后的复现都保存，收缩结果不能替代原始工件。

### 8.3 52 种变换的精确定义

下表中“第一/第二个”都指 canonical 稳定 ID 顺序，“第一个配置”指 JSON pointer 的 NFC UTF-8 bytes 字典序。新的占位 ID 从 `node.missing`、`edge.mutation`、`region.missing` 开始；若已存在，依次追加 `.0`、`.1`，取第一个未使用且通过稳定 ID 字法的字符串。carrier 前置条件是 corpus 生成 preflight；不满足时不得用另一种变换凑数。

除下表明说“留下 stale”的 integrity 变换外，每次变换后都执行同一 reseal：用对应 schema 的 canonical writer 重写已修改 record，重算受影响 Plan bytes/hash、typed record 中的 logical hash、parameter-schema 的 logical hash、Tensor manifest/hash 和最外层 content hash。reseal 只更新由主字段机械派生的认证字段，不重新生成参数、输入、状态或期望结果，因而不计为第二个语义变换。对非法 Plan，canonical writer 只排序/编码 raw JSON，不在 reseal 时调用 Plan validator。变换后除指定字段和这些派生字段外，与 base bundle 的 canonical decoded record exact 相等。

| mutation ID | carrier 额外前置 | 唯一主变换 |
| --- | --- | --- |
| `bundle-content-hash` | 无 | 把最外层 `content_hash` 的第一个 hex nibble 改为字典序最小的不同小写 hex nibble；不 reseal |
| `canonical-bytes-hash` | 无 | 同样只改最外层 `logical_plan_hash` 的第一个 nibble，logical Plan bytes 及 typed Plan 内部 logical hash 不变；只重算最外层 content hash，留下 bytes/hash 的唯一 stale 关系 |
| `bundle-root-version` | 无 | 把 fixture 根 `schema_version` 改为 `tide.settlegraph.fixture.invalid-v2`，其余按 reseal |
| `tensor-manifest-shape` | 至少一个非标量 Tensor | 把 Tensor manifest 中 JSON pointer 最小条目的最后一个 shape 维度加 1，不改 Tensor；重算 tensor-artifact 和 content hash，不从 Tensor 重建 manifest |
| `plan-root-version` | 无 | 把 logical Plan 根 `schema_version` 由 `1` 改为 `2` |
| `illegal-stable-id` | 至少一个 node | 只把第一个 node record 的 `node_id` 改为带前导 ASCII 空格的字面值 ` node.bad`，引用它的 region/edge/boundary 字段不改 |
| `duplicate-node-id` | 至少两个 nodes | 把第二个 node record 的 `node_id` 改为第一个的 ID，其他引用不改 |
| `duplicate-edge-id` | 至少两条 edges | 把第二条 edge record 的 `edge_id` 改为第一条的 ID |
| `cycle` | 存在分属不同 regions 的终端 $z$ 和入口 $a$，且已有 $a\leadsto z$ 路径、没有 $z\to a$ edge | 新增一条 ID 为首个可用 `edge.mutation[.n]`、label `data`、source $z$、target $a$ 的 edge |
| `intra-region-edge` | 同一 region 有两个 nodes $u<v$，且没有 $u\to v$ | 新增一条首个可用 mutation ID、label `data`、source $u$、target $v$ 的 edge |
| `unknown-edge-endpoint` | 至少一条 edge | 把第一条 edge 的 `target` 改为首个可用 `node.missing[.n]` |
| `wrong-entry-set` | 入口集至少一项 | 从 `entry_node_ids` 删除第一个 ID |
| `wrong-terminal-set` | 终端集至少一项 | 从 `terminal_node_ids` 删除第一个 ID |
| `foreign-state-owner` | 至少一个 stateful node $u$ 和另一 node $v$ | 把 ID 最小的 stateful node 的 `state_owner` 从 $u$ 改为 ID 最小的 $v\ne u$ |
| `formula-unknown-key` | 无 | 在 JSON pointer 最小的 operation config 中加入额外键 `unexpected` 和数值 0 |
| `formula-missing-required` | 至少一个 GDN Update | 从第一个 GDN Update config 删除无默认的 `key_dim` |
| `formula-id-type-mismatch` | 至少一个 stateless Update | 把第一个 type `none` Update 的 `formula_id` 从 `update.none.v1` 改为 `state.ema.v1`，type 及其他键不变 |
| `bias-false` | 存在 schema 固定 `bias=true` 的 operation | 把 JSON pointer 最小的该 config 的 `bias` 改为 false |
| `derived-output-shape` | 存在非标量 `output_shape` | 把 JSON pointer 最小的 operation config 的 `output_shape` 最后一维加 1 |
| `derived-state-shape` | 至少一个 stateful node | 把第一个 stateful Update config 的派生 `state_shape` 最后一维加 1，node 自身 `state_shape` 不变 |
| `n-pre` | 至少一个 N/content region | 把第一个该 region 的 `selector_timing` 改为 `pre` |
| `sd-post` | 至少一个 SD region | 把第一个该 region 的 `selector_timing` 改为 `post` |
| `fixed-k-zero` | 至少一个 fixed-K region | 把第一个该 region 的 `k_requested.value` 改为整数 0 |
| `values-by-node-keyset` | 至少一个 fixed-by-node Score 且 region 非空 | 从第一个该 Score 的 `values_by_node` 删除 node ID 最大的键 |
| `binding-missing-role` | 无 | 从 typed Plan `binding.dtype_roles` object 中只删除 `readout` role |
| `binding-symbolic-dtype` | 无 | 把 typed Plan `binding.dtype_roles.readout` 的 concrete dtype 字面值改为 `runtime` |
| `hidden-shape` | hidden 最后一维非空 | 用同 dtype 的 $[B,T,d+1]$ CPU Tensor 替换 `inputs.hidden`；前 $d$ 列 exact 拷贝 base，新列由对应 mutation 标签的 dyadic 域填充 |
| `hidden-dtype` | 无 | FP64 carrier 把 hidden 转为 FP32，FP32 carrier 把 hidden 转为 FP64；typed binding 不变 |
| `duplicate-sequence-id` | $B\ge2$ | 把 `inputs.sequence_ids[1]` 改为 `inputs.sequence_ids[0]` |
| `reset-id-duplicate` | 至少有一个 sequence ID | 把 ID 最小的 sequence 连续写两次作为 `control.reset_sequence_ids` 的全部内容 |
| `requested-k-region-keyset` | 至少一个 input-K region | 从 `control.requested_k` 删除 region ID 最小的键 |
| `requested-k-shape` | 至少一个 input-K region | 把 region ID 最小的 $[B,T]$ K Tensor 替换为 `unsqueeze(-1)` 得到的 $[B,T,1]$ Tensor |
| `execution-mask-shape` | 无 | 把 $[B,T]$ 的 `inputs.execution_mask` 替换为 `unsqueeze(-1)` 得到的 $[B,T,1]$ bool Tensor，值与 base 的唯一新尾维 slice exact 相同 |
| `execution-mask-dtype` | 无 | 把 `inputs.execution_mask` 的 shape/value 保持不变，只从 bool 转为 int64 |
| `token-position-shape` | 无 | 把 $[B,T]$ 的 `inputs.token_positions` 替换为 `unsqueeze(-1)` 得到的 $[B,T,1]$ int64 Tensor，值与 base 的唯一新尾维 slice exact 相同 |
| `token-position-dtype` | 无 | 把 `inputs.token_positions` 的 shape 和整数值保持不变，只从 int64 转为 FP64；所有值在 FP64 中精确可表示 |
| `lm-mask-outside-execution` | 存在 execution=false 且 LM=false 的位置 | 在 row-major 最小的该位置只把 `lm_target_mask` 改为 true |
| `routing-mask-outside-execution` | 存在 execution=false 且 routing=false 的位置 | 在 row-major 最小的该位置只把 `routing_stats_mask` 改为 true |
| `position-replay` | 某 row 至少两个连续执行位置 | 对 row ID/物理列字典序最小的该 pair，把第二个 `token_positions` 值改成第一个值 |
| `position-skip` | 与 replay 相同 | 对同样选定的 pair，把第二个位置值改成第一个值加 2 |
| `state-owner-key` | 至少一个 receiver-state entry | 把 canonical `initial_state.receiver_values` 第一项的 `node_id` 改为首个可用 `node.missing[.n]` |
| `state-shape` | 至少一个 Tensor receiver state | 把第一个 state Tensor 的最后一维加 1；原有元素按 row-major 前缀拷贝，新元素由 mutation dyadic 域填充 |
| `owner-alias-object` | 有两个 shape/dtype/value exact 相同的 Tensor receiver states | 把第二个 entry 的 value 设为与第一个 exact 同一 Tensor object，manifest 如实记录共享 storage；由于 carrier 值相同，此变换只新增 alias 关系 |
| `owner-alias-view` | 至少两个同 dtype Tensor receiver states | 建立一个长度为两者 `numel` 之和的 CPU backing Tensor，按 row-major 拷贝原值，再把两个不重叠、正 stride 的相邻 slices 分别 reshape 回各自原 state shape，以这两个 views 替换前两个 values；两者 shape/dtype/value exact 不变，manifest 如实记录同 storage group |
| `requested-k-zero-late` | 指定 input-K event 前至少两个成功 Token 且该 event candidates 非空 | 只把该 event 的 `requested_k` 整数改为 0 |
| `requested-k-above-max-late` | 与 zero-late 相同 | 只把该 event 的值改为该 region 规范 $K^{\max}+1$ |
| `local-update-failure-late` | 指定 stateful Update event 前至少两个成功 Token | 在 negative container 的唯一 `fault_hook` 字段写入 hook ID `fault.update.raise.v1` 和该 node event 稳定键；不改 Plan 或数值 |
| `local-node-failure-late` | 指定 active NodeCompute event 前至少两个成功 Token | 同上，hook ID 为 `fault.node-compute.raise.v1` |
| `empty-terminal-first` | 首个执行位置至少有一个 active terminal | 写入 hook ID `fault.terminal-send.suppress.v1`，target 为首个执行位置的 output event key，抑制该 event 的全部 terminal sends |
| `empty-terminal-late` | 指定 output event 前至少两个成功 Token 且该 event 有 active terminal | 写入同一 suppress hook，target 为满足前置的最小 output event key |
| `attention-window-order` | 某 Attention state 至少两个有效位置 | 对 owner key 最小的该 state，只交换 `positions[0]` 和 `positions[1]`，keys/values 不变 |
| `attention-window-length` | 某 Attention state 已有效长度恰为 Plan window | 在 owner key 最小的该 state 后追加一个 position 为原最大 position 加 1 的 key/value triple，key/value 由 mutation dyadic 域物化，保持位置升序但使长度为 window+1 |

`fault_hook` 是 `tide.settlegraph.negative.v2` 的测试字段，不是 logical Plan 或生产调用的新语义。loader 只校验其 schema 和 target event 是否按 expected trace 可达；runner 仅在对应资格构建中安装 hook。除表中四个 hook mutation IDs（共三种 hook IDs）外，`fault_hook` 必须 absent。

## 9. Chunk、mask、生命周期与 checkpoint scenarios

### 9.1 Scenario 工件

单调用 `tide.settlegraph.fixture.v1` 不足以表达完整生命周期。资格实现应增加独立 scenario 工件；建议 serialized ID 为 `tide.settlegraph.scenario.v1`，但只有 schema、loader 和 tests 实际落地后才能在 README 中称其存在。它至少保存：

- scenario ID、源 bundle hashes 和有序 `actions`；
- 每个 action 的输入、预期结果或 failure envelope；
- action 前后的公开 state、next-position、RNG 和 artifact-set hashes；
- call 的三个 masks、row 到 `sequence_id` 映射、全局位置、K、detach 和 trace policy；
- fault/barrier 的稳定测试 hook ID；
- checkpoint action 的输入/输出 hash 和 fresh-process identity。

测试 hook 只能在资格构建中启用，不能改变无故障公式。

### 9.2 固定 scenario 集

| 集合 | 确定性输入 | 必须执行的变体 |
| --- | --- | --- |
| short chunk | 六个 margin-safe legal bundles，分别令 $T=1,2,3,4,5,6$；$T\ge2$ 的五个必须 stateful，且冻结的后块标量目标对第一个边界前的至少一个 state/input key 有非零跨边界 VJP | 对每个 $T$ 枚举全部 $2^{T-1}$ 个边界子集，共 63 种切法；$T=1$ 没有边界，只比较 forward/trace/state；$T\ge2$ 的每个非空切分同时执行 detach 和 no-detach，要求 forward 相等且冻结的跨边界 VJP 分别断路/连通 |
| long chunk | 四个 $T=17$ bundles，覆盖 none/EMA/GDN/Attention | 单 chunk、逐 Token、长度交替 1/2、`[5,4,3,2,1,2]` 加空尾调用 |
| masks/rows | 八个 $B=3,T=5$ bundles | prompt 无 LM target、padding tail、routing-stat 子集、不等长 rows、下一调用 row 重排 |
| lifecycle | 八个 stateful bundles | fresh create、连续调用、site-local reset、单 site 下的 all-site reset、release 后重建；release 为幂等成功，第二次结果固定为 `status=ok, released=[], already_absent=[<sorted requested IDs>]` |
| position negatives | 六个 scenarios | replay、倒序、跳号各两个；任何执行前失败 |
| controlled concurrency | 四个 scenarios | 两个 call 写同一 sequence 的相邻、不重叠 positions；barrier 固定 A 先获得 sequence lease，B 在 A publish 前请求并等待；A publish 后 B 重新校验并执行，两者都成功，结果 exact 等于 A→B 的唯一串行顺序 |
| late rollback | 16 个 scenarios | late K 4、late Update failure 4、late NodeCompute failure 4、late empty terminal 4 |

single-site 中 site-local reset 与 all-site reset 的结果应相同，但这不构成多 site reset 证据。release 只能针对没有进行中调用的明确 sequence IDs；scenario action 要求请求 IDs 已经唯一并按稳定字符串排序，不存在的 ID 不是 failure。受控并发唯一允许的资格结果是上表 A→B 串行化；不再保留“或拒绝一个 call”的运行时选项。它不依赖 wall-clock 谁先到达，而使用测试屏障固定 lease 获取顺序。

完整与分块执行比较逐 Token output、exact trace 语义 events、最终规范 state、next position、$(N,P_v,A_v,F_v,Q)$、最终 balance loss 和适用 VJP。充分统计先相加再 reduction；不得平均 chunk losses。

### 9.3 Core checkpoint

16 个 optimizer fixtures 各建立一个 checkpoint scenario：

1. fresh process A 执行两个 calls 和一次 optimizer step，在 detach 且统计窗口边界保存；
2. A 不退出地继续下一个 call 和 optimizer step，保存参考结果；
3. fresh process B 用 `resume` 加载并执行同一下一步；
4. fresh process C 用 `init-from` 加载，证明只恢复声明的权重，optimizer、CPU RNG、进度、receiver state 和位置重新开始；
5. 一个 late-failure call 后立刻保存，证明 checkpoint 看不到失败调用的 staged state。

`core-v1` checkpoint cell 只声明当前 schema 明确保存的 SettleGraph 参数、Adam/AdamW state、CPU RNG、receiver state、next position和训练 metadata。scheduler、AMP scaler、backend RNG、sampler/data cursor、未归约统计窗口和跨保存点 autograd 图没有实现前，不得把这里的通过称为完整训练 exact resume。

checkpoint negatives 在第 8 节的 104 个 negative artifacts 之外另做下表 16 个。按表内 ID 顺序，从 16 个 optimizer fixtures 求字典序最小的一对一 carrier 排列；`ckpt-compat-state-shape` 的 carrier 额外要求至少有一个非空 Tensor receiver state，其他 ID 无额外前置。无法形成这个排列时 corpus preflight 失败。每个 scenario 都物化 FP64/FP32 子 cell，不计入第 8 节的 96 个 Plan/运行期输入或 8 个 artifact instances。除 integrity 故障是修改被认证 bytes 外，其他 mutation 都重算外层 hash，以到达表中的唯一最早 phase。

| expected code | 固定 checkpoint negative IDs |
| --- | --- |
| `checkpoint.integrity` | `ckpt-integrity-content-hash`、`ckpt-integrity-tensor-hash`、`ckpt-integrity-truncated-bytes`、`ckpt-integrity-unsafe-object` |
| `checkpoint.schema` | `ckpt-schema-root-version`、`ckpt-schema-missing-root-key`、`ckpt-schema-extra-root-key`、`ckpt-schema-wrong-root-type` |
| `checkpoint.compatibility` | `ckpt-compat-logical-plan`、`ckpt-compat-typed-binding`、`ckpt-compat-optimizer-kind`、`ckpt-compat-state-shape` |
| `checkpoint.commit` | `ckpt-commit-model`、`ckpt-commit-optimizer`、`ckpt-commit-cpu-rng`、`ckpt-commit-sequence-state` |

四个 commit 注入分别落在 model、optimizer、CPU RNG 和 sequence state 发布处，每个都要求这四类对象联合回滚。`ckpt-compat-state-shape` 只把按稳定 owner key 排序的第一个 receiver Tensor state 的最后一维加 1，不同时改 owner 或 dtype。

runtime negatives 另做下表 8 个。前四个只调用 CPU-safe request normalization；后四个在 fresh process 中使用资格专用、冻结的 availability provider 驱动生产 resolver 的同一分支，provider record/hash 进入工件，不依赖测试主机恰好缺少某个硬件。

| expected code | ID | 唯一请求/故障 |
| --- | --- | --- |
| `runtime.configuration` | `runtime-config-unknown-device` | `device="tpu"` |
| `runtime.configuration` | `runtime-config-duplicate-index` | `device="npu:0"` 且另给 `device_index=0` |
| `runtime.configuration` | `runtime-config-negative-index` | `device="npu"`, `device_index=-1` |
| `runtime.configuration` | `runtime-config-auto-index` | `device="auto"`, `device_index=0` |
| `runtime.unavailable` | `runtime-unavailable-backend` | 显式 `npu:0/float32`，provider 报告 plugin/device 不可用 |
| `runtime.unavailable` | `runtime-unavailable-index` | provider 报告仅 1 个可见 NPU，显式请求 logical index 1 |
| `runtime.unavailable` | `runtime-unavailable-dtype` | provider 接受合法 `npu:0/bfloat16` 请求，但 minimum probe 精确报告该 dtype 不可用 |
| `runtime.unavailable` | `runtime-unavailable-operator` | provider 接受 `npu:0/float32`，但 minimum `elementwise-square-forward-backward` probe 精确报告 required operator 不可用 |

所有 runtime negatives 必须非零失败，不能 skip、切到 `auto` 或回到 CPU。availability provider 不能直接构造 expected envelope；actual code 仍由 runtime request/resolver 的已知阶段捕获。

## 10. 比较器和资格 cells

### 10.1 共同阈值

所有浮点比较先在产生结果的 backend 检查 finite，再复制到 CPU float64 计算逐元素误差。对 reference $x$ 和 candidate $y$，要求

$$
|y_i-x_i|\le\mathrm{atol}+\mathrm{rtol}|x_i|.
$$

| 阈值名 | `atol` | `rtol` | 适用范围 |
| --- | ---: | ---: | --- |
| `T64` | $10^{-10}$ | $10^{-8}$ | CPU FP64 小 fixture |
| `T32` | $10^{-6}$ | $10^{-5}$ | 同一 backend FP32 executor |
| `TN32` | $10^{-4}$ | $10^{-4}$ | CPU FP32 与 NPU FP32 |
| `TFD64` | $10^{-6}$ | $10^{-4}$ | 第 7.1 节的方向有限差分标量 |

schema、hash、keys、shape、声明 dtype、mask、owner、event key、candidates、K、route、Top-K IDs、reached/Observe/active/send、edge status、Attention positions、failure envelope 和 checkpoint key set 全部 exact。所有 trace 还必须独立通过等价性测试契约第 4.1 节的单边 invariants；两个 executor 共同产生同一错误不能通过。

每个浮点 comparator artifact 保存量的稳定路径、最大绝对/相对误差、最坏 reference/candidate 值和阈值，不能只保存一个全局 bool。

### 10.2 Cell 表

表中“共同证据”指第 2.3 和 3.2 节的 manifest、stdout、terminal summary、case list、hash list 与 comparator worst-path records。任何 required case 失败、缺工件、被 skip 或数量不足，整个 gate 失败。下表每一行是 gate template，表中数量是该行全部子 cells 的合计；实际 capability cell ID 必须追加 host architecture、backend、单一 dtype 和 executor binding，并在行级 summary 中引用。例如 `C03` 展开为 FP64 和 FP32 子 cells，各含 256 个 materialized cases；`C11` 两个 dtype 子 cells 各含 52 个 mutants，行级总数为 104，其中非 artifact 的 96 个实例才贡献上位契约的非法 Plan/运行期输入门槛。只有 evidence I/O 的 `I00` 不分数值 dtype，只有 logical corpus identity 的 `C00` 使用明示 dtype set；两者都不由此声称数值 executor 能力。x86_64 CPU、aarch64 CPU 及不同 NPU SKU 永远不合并成一个 capability cell。

| Cell | 输入 | 执行器/环境 | 比较量 | 阈值 | 必需工件 | 通过条件 |
| --- | --- | --- | --- | --- | --- | --- |
| `I00-evidence-infrastructure` | 第 3.3 节固定 35 probes | CPU-safe schema/manifest/fixture publisher/run-record 路径；I/O 故障用一次性 hook 和 fresh process | failure phase/code、primary/secondary identity、exit status、case IDs、inode/path set、published bytes/hash、Tensor storage hash、gradient key set、module/runtime identity、terminal records | exact | probe manifest、fault timeline、发布前后目录/inode/hash、runner summaries | 35/35 全部符合预期；无 raw 类型/重叠异常泄漏、无 hash 类型碰撞或未认证 storage、无半发布、无 primary 被遮蔽、无外部代码阴影、无 skip/缺 ID/预期失败；其他 cell 引用本门的通过 identity |
| `C00-canonical` | 256 legal 的 logical/typed bytes、104 negative containers、默认省略/显式成对输入 | CPU-safe canonicalizer、negative-container parser 和静态/loader 入口；不导入 vendor plugin | canonical bytes/hash、默认物化、parameter manifest、bundle hash、negative-container schema；只对声明静态/loader 入口的 mutant 比较 actual envelope | exact | 共同证据、canonical goldens、所有 schema/preflight reports | 两次独立生成 byte-identical；256 正例通过；104 containers 的 schema 全部合法；92 个静态/loader mutants 在预期最早 phase 失败，12 个 event/execution mutants 通过 loader 并留给 `C11` 触发 |
| `C01-golden-forward` | 12 legal goldens，FP64/FP32 | token-major eager CPU | 全部 exact trace、output/state/statistics、invariants，以及 `(field,type,formula_id)` 独立 oracle coverage | `T64`/`T32` + exact | 共同证据、手工推导、expected/actual traces、`golden-formula-coverage.json` | 24 个 materialized runs 零失败；core-v1 registry uncovered list 为空；期望代码未调用共享局部 helper |
| `C02-golden-fault` | 2 fault goldens，两 dtype | token-major eager CPU + test hook | failure envelope、调用前后公开 state/RNG/artifact set | 浮点不发布；其余 exact | 共同证据、private diagnostic trace、before/after hashes | 四次都在指定 phase/code 失败且整调用回滚 |
| `C03-reference-forward` | 全部 256 legal，两 dtype；每个 materialization 的 A/B 输入 bytes 相同 | 固定线程/确定性设置下的两个 fresh-process token-major eager CPU replay | A/B output/state/statistics/full trace；bundle 预期 route、invariants 和 event coverage | A/B 浮点用 `T64`/`T32`；离散量 exact | 共同证据、256 family 的 A/B results、coverage reports、已通过的 `C01` 身份 | `C01` 先通过；512 个 materialized cases、1024 次 process executions 零失败；256 logical hashes 唯一；所有数量与 pairwise 门槛通过 |
| `D00-region-major-development` | 全部 256 legal，两 dtype | token-major 对 region-major eager CPU | 与 `C03` 相同 | `T64`/`T32` + exact | 共同证据、差分 records | 零差分只表示开发回归；无论结果如何都不得晋级 packed cell |
| `C04-packed-forward` | 全部 256 legal，两 dtype | token-major reference 对通用 packed CPU | output/state/statistics/full trace、route、invariants、dtype | `T64`/`T32` + exact | 共同证据、packed schedule identity、差分 records | 512 对零失败；profile/instrumentation 证明热路径没有逐 Token/row/node Python 调度 |
| `C05-specialized-forward` | 256 中静态支持谓词为 true 的全部 cases | token-major、packed、对应特化 CPU 三方 | `C04` 的全部量 | `T64`/`T32` + exact | 支持谓词版本、accepted/rejected lists、三方 records | 所有预先适用 case 零失败；不适用 case 在构造前稳定拒绝；不能按结果改列表 |
| `C06-vjp` | 固定 64 VJP，两 dtype；另冻结 32 个 FP64 FD；8 个 HST 和 8 个 chunk objective 各物化两 dtype | token-major 对 packed CPU；独立标量差分、HST 解析局部 VJP 和 detach 路径 oracle | objective、每个 logical key 的连接类别和 VJP、state/input grads、HST 局部导数、chunk forward-equal/backward-cut | `T64`/`T32`、32 项另用 `TFD64` | 共同证据、objective/cotangent、全 gradient records、FD/HST/chunk records | 128 executor pairs、32 FD、16 HST materializations 和 16 chunk materializations 零失败；没有未声明 `None` 或 nonfinite |
| `C07-optimizer` | 固定 16 optimizer，两 dtype | token-major 与 packed 各自 backward + Adam/AdamW step | step 前 grads、step 后全部参数、optimizer Tensor/state/key order | `T64`/`T32` + exact metadata | 共同证据、初始/最终 parameter 与 optimizer snapshots | 32 对零失败；8 Adam、8 AdamW 及 cold/warm 数量 exact |
| `C08-chunk` | short/long chunk scenarios | token-major 与 packed CPU，full 对所有 split | 逐 Token output/trace、state、position、stats/loss、detach/no-detach VJP | `T64`/`T32` + exact | scenario、每个 action result、统计合并 records | 第 9.2 节所有切法零失败；空尾无事件；统计只先加后归约 |
| `C09-mask-lifecycle` | masks/rows/lifecycle/concurrency scenarios | token-major 与 packed CPU | bypass、state owner、row reorder、reset/release、冲突结果 | `T64`/`T32` + exact | scenario、action timeline、before/after hashes | 所有合法动作等价；所有非法动作发布前失败；single-site 声明不外推多 site |
| `C10-rollback` | 固定 16 late rollback scenarios，两 dtype | 两 CPU executors + test hook | envelope、公开 state/position/RNG/artifact set、无部分 result | exact | 原始/收缩 fixture、private trace、before/after hashes | 64 executor runs 都在预期点失败并 exact rollback |
| `C11-invalid` | 固定 104 mutants（96 个 Plan/运行期输入，8 个 artifact） | 独立 validator/loader；随后对到达 execution 的 case 运行 token-major 和 packed | actual phase/full code set、failure priority、无发布 | exact | negative containers、actual envelopes、原始与 shrink artifacts | 104 个各自得到唯一预期 envelope；actual code 不来自 expected 注入；上位 96 门槛不计 artifact faults |
| `C12-checkpoint` | 16 positive、16 checkpoint negative、8 runtime negative scenarios | CPU fresh processes；token-major 与 packed binding 各自加载 | forward/state/VJP、下一 optimizer step、init/resume 边界、联合 rollback | `T64`/`T32` + exact schema | checkpoint hashes、process manifests、before/after snapshots | 所有 positive 通过，所有 negative 精确失败；只声明第 9.3 节保存范围 |

`C04`、`C06` 至 `C12` 中要求 packed 的部分，在 packed executor 或 scenario artifact 尚未实现时状态只能是 `planned`。可以先运行其 token-major/region-major 投影以发现问题，但不能删掉 packed 比较后沿用同一个 cell ID。

## 11. NPU 的预冻结 set-cover 子集

### 11.1 选择算法

NPU 只消费已经通过 CPU artifact preflight 的 byte-identical FP32 bundles。任何 NPU case 执行前，`npu-subsets.json` 按下列顺序冻结三个嵌套集合：

$$
S_{\mathrm{opt},8}\subseteq S_{\mathrm{vjp},32}\subseteq S_{\mathrm{fwd},64}.
$$

coverage universe 包括：

- 每个声称支持的 formula 的 forward，以及训练公式的 backward；
- 六种 profile/timing、四种 state、三种 receiver Aggregate、两种 output Aggregate、三种 Emit、两种 K source 和四种 route class；
- B=1、T=1、奇数 hidden/state dimension、最小/最大窗口、空 candidate event、padding/空段、packed 尾块和 noncontiguous layout；
- sort/Top-K、mask/nonzero、count/cumsum、gather/scatter、`index_add`、归约、linear、normalization 和 Attention 的实际 shape/layout/direction units；
- Adam、AdamW、cold/warm optimizer state 和 CPU checkpoint handoff。

这个 universe 在 registry 中预先分为三个不重叠层级：$U_{\mathrm{opt}}$ 含 optimizer/checkpoint handoff 单元，$U_{\mathrm{vjp}}$ 含公式 backward 和 backward operator/shape/layout 单元，$U_{\mathrm{fwd}}$ 含其余 forward、route、state、mask、shape 和 layout 单元。一个 case 对某个 unit 的 coverage 只能来自 CPU preflight 中实际执行该 direction 所产生的 event/operator record，不能仅根据 Plan 声明推断。

先在 16 optimizer candidates 上贪心选 8 个，目标是覆盖 $U_{\mathrm{opt}}$；每步选择对当前层级尚未覆盖 units 的 gain 最大的 case。然后以这 8 个为初始集，从 64 VJP candidates 中按同一规则补到 32，目标是覆盖 $U_{\mathrm{opt}}\cup U_{\mathrm{vjp}}$；再从 256 legal 中补到 64，目标是覆盖三层 universe 的并集。覆盖数相同时，依次比较新覆盖 unit 的稀有度总权重和 fixture ID。每个 unit 的权重在 selection manifest 中预先固定为 `1/count_in_cpu_corpus`，不能看 NPU 运行结果。

若任一层到达固定大小后仍有该层目标 unit 未覆盖，selection preflight 失败；不得增加数量、删除 unit 或用运行后通过的 case 替换失败 case。集合成员、每步 gain、完整 covered/uncovered units 和 manifest SHA-256 必须在首次 NPU 进程启动前保存。

### 11.2 NPU cells

每个 case 在只导入 NPU vendor family 的 fresh process 中解析显式 `--device npu --device-index LOGICAL_INDEX --dtype float32`。物理可见性由 launcher 设置，进程记录的 index 是 remap 后逻辑 index。显式 NPU 不可用时是 cell failure，不是 skip。

| Cell | 输入 | 执行器/环境 | 比较量 | 阈值 | 必需工件 | 通过条件 |
| --- | --- | --- | --- | --- | --- | --- |
| `N00-runtime-operators` | 从 64 集合派生的全部 operator/shape/layout probes | NPU eager，真实 allocation、operation、sync、CPU copy | 数值、shape/layout、forward/backward、empty/tail behavior | `TN32` + exact | runtime manifest、probe results、stderr/warnings、dispatch/profile | 全部 required probes 实际位于 NPU；无未声明 CPU fallback |
| `N01-eager-forward` | 固定 64 FP32 bundles | CPU result artifact 对 NPU token-major eager | output/state/statistics/full trace、自然 route、invariants | `TN32` + exact route | 共同证据、64 result artifacts、placement evidence | 64/64 零失败；每个 case fresh process；coverage universe 完整 |
| `N02-eager-vjp` | 固定嵌套 32 | CPU VJP artifact 对 NPU eager VJP | objective、所有声明 gradient keys/连接类别 | `TN32` + exact metadata | 32 gradient artifacts、backward profiles | 32/32 零失败、finite，关键 backward ops 无 fallback |
| `N03-eager-opt-checkpoint` | 固定嵌套 8 | CPU checkpoint 到 NPU eager portable handoff | step 前 grad、下一 step 参数/optimizer state、state continuation | `TN32` + exact schema | 8 CPU checkpoints、8 NPU snapshots、profiles | 8/8 零失败；只声明 portable handoff，不声明跨 backend exact resume |
| `N04-packed-forward` | 同一固定 64 | CPU artifact、NPU eager、NPU packed 三方 | `N01` 全部量及 packed schedule | `TN32`，同 NPU eager/packed另用 `T32` | 三方 records、packed profiles | 64/64 零失败且真实 packed；region-major eager 不能代替 |
| `N05-packed-vjp` | 同一固定 32 | CPU、NPU eager、NPU packed | `N02` 全部量 | `TN32`/`T32` + exact | 三方 gradient records、profiles | 32/32 零失败且 backward 无 fallback |
| `N06-packed-opt-checkpoint` | 同一固定 8 | NPU packed + CPU handoff | `N03` 全部量 | `TN32` + exact schema | parameter/optimizer/checkpoint snapshots、profiles | 8/8 零失败；optimizer 和 load 后首步均有 placement closure |

正确数值或一个 NPU output Tensor 不足以通过。每个关键 operator 必须有当前 TorchNPU/CANN 栈可解释的 profiler/dispatch 证据；若 fallback 可见性未知，相应 cell 保持 `implemented`，不能标 `verified`。x86_64/aarch64、不同 Ascend SKU、eager/packed 和每个软件栈 tuple 都是不同 cells。

## 12. 独立门：HB Builder、Qwen、短训练和性能

这些门不贡献 256 legal、64 VJP、16 optimizer、96 Plan/运行期输入 negative 或另计 8 个 artifact negative 的 core executor 数量。每个门只有在其输入 identity 冻结后才能运行。

### 12.1 冻结的性能 workload

`performance-workloads.json` 固定下表 12 行，不把“代表组合”留给 benchmark runner 临时选择。每行先从第 5 节的 legal families 中取同时满足 topology 和 probe state 两列的最小 fixture ID，再作下列唯一性能输入物化：

- logical Plan 保留该 family 的全部拓扑和公式；每个非 singleton region 的 active budget 规范化为 `k.input.v1`，`maximum` 保留该 region 的 $K^{\max}$，singleton 仍是 forced-active fixed 1。改写后的 canonical bytes/hash 单独冻结，不沿用源 family hash。
- `one`、`half` 和 `all` 分别令每个非 singleton region 事件的 `requested_k` 为 $1$、$\lceil K^{\max}/2\rceil$ 和 $K^{\max}$；它们表示请求密度，实际 active 数仍按语义取候选数与请求值的较小者。
- hidden 是 contiguous，数值由 `x("perf:<workload_id>:hidden", i)` 产生；状态 fresh，execution/routing-stat masks 全 true，LM target mask 全 false，每行使用独立稳定 sequence ID，positions 从 0 连续增加。

| workload ID | topology | probe state | B | T | K 请求密度 |
| --- | --- | --- | ---: | ---: | --- |
| `perf-00` | `single-layer-r8` | `none` | 1 | 128 | `one` |
| `perf-01` | `single-layer-r8` | `ema` | 8 | 128 | `half` |
| `perf-02` | `single-layer-r8` | `gdn` | 1 | 512 | `half` |
| `perf-03` | `single-layer-r8` | `attention-window` | 8 | 512 | `all` |
| `perf-04` | `chain` | `none` | 1 | 2048 | `all` |
| `perf-05` | `chain` | `ema` | 8 | 2048 | `all` |
| `perf-06` | `small-hb` | `gdn` | 1 | 128 | `one` |
| `perf-07` | `small-hb` | `attention-window` | 8 | 128 | `half` |
| `perf-08` | `small-hb` | `none` | 1 | 512 | `all` |
| `perf-09` | `generated-dag` | `ema` | 8 | 512 | `one` |
| `perf-10` | `generated-dag` | `gdn` | 1 | 2048 | `half` |
| `perf-11` | `generated-dag` | `attention-window` | 8 | 2048 | `all` |

这个选择和改写在任何 correctness 或 timing 进程前完成。若某行没有满足条件的 legal family，或改写后 Plan 无法通过 validator，performance preflight 失败；不得换用运行后更快的 family。12 行分别保存 source family ID、新 logical/typed Plan hash、input/state hash 和完整 formula/operator event 列表。

### 12.2 专项 cell 表

| Cell | 输入 | 执行器/环境 | 比较量 | 阈值 | 必需工件 | 通过条件 |
| --- | --- | --- | --- | --- | --- | --- |
| `X00-hb-builder` | 固定 Builder name/version/config 列表及 expected endpoints/labels/hash | CPU Builder + canonicalizer | fully expanded Plan bytes、endpoint、Line/phase/barrier、edge labels | exact | configs、expanded Plans、golden hashes | 每个 config exact；只声明列出的 Builder 域 |
| `X01-qwen-identity-cpu` | pinned Qwen/model/tokenizer revision；4 placements × prefill/decode，共 8 fixtures | 原 Base 对单-site接入模型，CPU FP32 | hidden/logits、causal mask、position IDs、KV cache、LM loss、共同 base/input grads | `T32`；mask/IDs exact | model lock、inputs、KV artifacts、gradient records | 8/8 通过；balance 差异关闭；只比较声明的共同参数梯度 |
| `X02-qwen-nonidentity-cpu` | 同一 lock；4 placements × prefill/decode，共 8 fixtures | CPU token-major 对真正 packed 的接入模型 | final logits、LM loss、KV continuation、SettleGraph state/route、base 与 graph grads | `T32` + exact route | 完整 Base fixture、traces、VJP records | 8/8 通过；decode最后位置与相同前缀 prefill 一致 |
| `X03-qwen-identity-npu` | 与 `X01` byte-identical 的 FP32 CPU inputs/weights | 原 Base 对单-site接入模型，NPU FP32；另读 CPU result | `X01` 全部量及 CPU/NPU parity、placement/fallback | 同 NPU 内 `T32`；跨 backend `TN32`；mask/IDs exact | NPU manifests、results、KV/grads、profiles | 8/8 通过，关键 operations 无 fallback；独立于 CPU cell 记录精确 SKU |
| `X04-qwen-nonidentity-npu` | 与 `X02` byte-identical 的 FP32 CPU bundles | NPU token-major 对 NPU packed；另读 CPU result | `X02` 全部量及 CPU/NPU parity、placement/fallback | 同 NPU 内 `T32`；跨 backend `TN32`；route exact | NPU traces、VJP records、profiles | 8/8 通过；真实 packed，decode/prefill 一致，无 fallback |
| `X05-short-train-cpu` | 冻结的 8 条 toy sequences、seeds 17/23/47、同一 Qwen lock | CPU FP32 packed，128 optimizer steps，step 64 resume | finite loss/grads、原始曲线、token accuracy、resume 下一步 | 每 seed 最后 16 step 平均 loss ≤ 最初 16 step 的 0.70；train token accuracy ≥0.90；resume 用 `T32` | run/metrics/summary、step-64 checkpoint、data/order hashes | 三个 seed 全部达标，无 nonfinite/skipped step，fresh-process resume 通过 |
| `X06-short-train-npu` | 与 `X05` byte-identical data/init，三 seeds | NPU FP32 packed，128 steps；CPU checkpoint handoff | 同 `X05`，另查 placement/fallback；不比较长期逐 step exact 轨迹 | 同任务阈值；handoff 首步 `TN32` | durable runs、raw metrics、profiles、handoff artifacts | 三个 seed 全部达标，关键 ops 无 fallback；不声称 CPU/NPU 长轨迹 exact |
| `X07-performance-cpu` | 第 12.1 节冻结的 `perf-00`–`perf-11` FP32 materializations | CPU FP32 token-major 与 packed；20 warmup、100 samples、5 fresh processes | synchronized p50/p95、throughput、peak memory、hot-loop Python callbacks | 长 $T\ge512$ 的 throughput 几何均值 ≥1.25×；每项 ≥0.95×；p95 ≤1.10×；peak memory ≤1.25× | raw samples、host/thread/NUMA manifest、profiles | correctness cells 先通过；无逐 Token/row/node Python hot loop；全部性能阈值通过 |
| `X08-performance-npu` | 与 `X07` byte-identical 的 `perf-00`–`perf-11` FP32 inputs/Plans | NPU eager 与 packed，计时前后同步 | `X07` 指标、device memory、fallback | 同 `X07`，阈值在首次运行前冻结 | raw samples、完整 stack/SKU、profiles | NPU correctness 先通过；12/12 的全部阈值和无 fallback 通过 |

Qwen lock 至少保存 immutable model revision、config/tokenizer hashes、输入 Token 和权重 identity；未选择具体 revision 前 `X01` 至 `X04` 保持 `planned`。四种 placement 是 POST、PARBLK、PARATTN 和 PARMLP。多 site 接入不从这些单-site fixtures 推断，必须等 schema v2。

短训练是功能和可学习性门，不是科学质量结论。`metrics.jsonl` 保存未平滑 loss、accuracy、learning rate、gradient norm、tokens/second 和 memory；Trackio 只能作为可重建的可视化投影，本地 record 才是证据。performance 的 timed region 包含 packing、状态读写、selector、NodeCompute 和 scatter，排除 bundle load、模型构造和一次性 compile；每个 sample 前恢复同一预分配状态，并在计时外完成该恢复。任何阈值变更都要求新的 workload contract 版本，不能看完结果再改。

Dense、Dense 扩展和 Flat MoE 若用于科学实验，另建各自的 reference、capacity/drop/reroute、loss、gradient、checkpoint 和性能门；它们不借用 `core-v1` 通过状态。

## 13. Schema v2 进入条件

extension fixture 只有在下表的决定被版本化并有 validator/canonical golden 后，才能从 `planned` 变为可执行：

| 扩展 | schema v2 必须先定义的内容 | 新增资格重点 |
| --- | --- | --- |
| selector-history | site/region/node owner 选择、规范 key、首值/decay、Read 维度、写回 stop-gradient、trace/state/checkpoint 表示 | node-level active EMA 的解析 trace、跨 Token/chunk/reset、failure rollback、VJP 断路 |
| 共享只读参数 | group ID、成员 logical keys、部分/整公式共享规则、formula/shape/dtype兼容、跨 site 命名 | 多使用点 forward、同一 Tensor identity、梯度求和、optimizer 只更新一次、checkpoint round trip |
| 固定/可学习首状态 | zero/fixed/learnable kind、owner 与 logical parameter key、shape/dtype、序列 materialization、reset、init-from/resume | 零/非零/reset、多个 sequences 的梯度累加、参数 manifest、optimizer 和 checkpoint |
| 多 site | stable site ID、parameter/state/trace key、顶层 reset/release/transaction、placement 顺序 | 四 placements、多 site 隔离、all-site reset、共享/非共享组合 |
| 低精度 | 每个 dtype role、accumulation/reduction dtype、rounding/autocast 和逐公式 tolerance | FP16/BF16 forward/backward/optimizer、overflow、CPU/NPU parity |
| adaptive budget | budget 输入来源、读取时点、值域、梯度、状态依赖和失败事务 | Tensor-derived K 的 forward/VJP、边界、chunk 与 replay |

schema v2 不能只增加可选字段而沿用 v1 hash 含义。它必须使用新 schema/canonicalizer identity，定义 v1 到 v2 是否存在无损升级，并为未知/缺失/冲突字段增加 negative fixtures。extension 的 256 legal、64 VJP、16 optimizer 和至少 96 个非法 Plan/运行期输入数量另立 corpus，不能把 `core-v1` 的事件重复计入；artifact faults 仍按其版本化工件 schema 另计。

## 14. 执行顺序和最终报告

资格执行按以下 gate 顺序进行：

1. 在相同 host architecture/commit 上先执行 `I00`，证明 schema、发布和 run-record 路径可承载资格证据；
2. 冻结 corpus、axes、legal pairs、VJP/optimizer 和 NPU subsets；完成 hash preflight；
3. `C00`、人工 golden、`C03` reference 和 `C11` invalid；
4. 真正 packed 落地后执行 `C04`，再执行 VJP、optimizer、scenario 和 checkpoint；
5. CPU cell 在 exact host architecture/commit 上全部完成后执行 NPU eager，再执行 NPU packed；
6. 独立执行 HB Builder、Qwen、short train；
7. 只有相关 correctness cells 全部通过后执行 performance。

最终报告逐 cell 列出 `planned`、`implemented`、`verified` 或 `unsupported`，并引用确实存在的 hashes 和相对 artifact paths。报告还必须列出：

- `I00` 35 个 probes 的 exact IDs、发布/终态故障时序和 primary/secondary 记录；
- 256 legal、64 VJP、16 optimizer、96 Plan/运行期输入 mutants 与另计 8 个 artifact mutants 的 exact IDs 和数量；
- axes、legal/covered/uncovered pairs 及 event multiplicities；
- 每个 dtype/backend 的最坏稳定路径误差；
- trace、VJP、optimizer、chunk、rollback 和 checkpoint 结果；
- NPU operator placement、warning 分类和 fallback closure；
- 未进入本次声明的 schema v2、Qwen、多 site、低精度、distributed、compiled/custom 和 baseline cells。

任一 required artifact 缺失时状态不能是 `verified`。NPU、CUDA、另一 host architecture、packed、Qwen 或训练没有实际运行时，应直接写未验证，不能由代码审查、CPU 结果或旧 commit 的 attempt 外推。

## 15. 现有 development corpus 的位置

当前 48 legal、6 VJP、24 invalid 集合只用于快速 development regression。它可以发现明显的 schedule、formula 或 validator 回归，但不满足本文的固定 bundles、独立多 motif goldens、pairwise、event multiplicity、64 VJP、16 optimizer、96 Plan/运行期输入 mutants、另计 8 个 artifact mutants、scenario、checkpoint 和硬件证据要求。

扩展该集合的 case 数、在本机跑通 region-major 或保存一次 NPU attempt，都不会自动把它变成资格 corpus。资格 runner 必须消费本计划冻结的独立 corpus identity，并生成第 2.3 节所述的 exact-commit terminal evidence。
