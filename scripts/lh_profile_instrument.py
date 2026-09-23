"""Optional exclusive calling-thread timers for an owned CROSSBATCH LH copy.

Applied after work counters; original arithmetic and scheduling are retained.
Unsupported revisions fail on unique anchors instead of silently misprofiling.
"""


def instrument(cpp):
    modified = []

    def change(name, edits):
        path = cpp / name
        text = path.read_text()
        for before, after in edits:
            if text.count(before) != 1:
                raise ValueError('profile anchor mismatch: ' + name + ': ' + before)
            text = text.replace(before, after)
        path.write_text('#include "tide/operator_profile.h"\n' + text)
        modified.append(path)

    change('src/AccumulateLocal.cpp', [
        ('{\n    const Tensor &x = bi.x;\n    const Tensor &sampleidsT',
         '{\n    tide::op_profile::Scope profile(tide::op_profile::InputPack);\n    const Tensor &x = bi.x;\n    const Tensor &sampleidsT'),
        ('    tide::work::linear(tide::work::QkvCalls',
         '    profile.phase(tide::op_profile::Qkv);\n    tide::work::linear(tide::work::QkvCalls'),
        ('    Tensor output;\n    if (KVHidden::attention_mode',
         '    profile.phase(tide::op_profile::StateOther);\n    Tensor output;\n    if (KVHidden::attention_mode'),
        ('    output = this->confluence.forward(output, nonzeros, indptr, I, sumcoe);',
         '    profile.phase(tide::op_profile::Pooling);\n    output = this->confluence.forward(output, nonzeros, indptr, I, sumcoe);'),
        ('    tide::work::linear(tide::work::OutCalls',
         '    profile.phase(tide::op_profile::Output);\n    tide::work::linear(tide::work::OutCalls')])
    change('src/BatchHidden.cpp', [
        ('void BatchPtrKVHidden::decay(double decay_rate)\n{',
         'void BatchPtrKVHidden::decay(double decay_rate)\n{\n    tide::op_profile::Scope profile(tide::op_profile::KvBuild);'),
        ('const Tensor &Icpu, const Tensor &SI)\n{\n    int64_t N',
         'const Tensor &Icpu, const Tensor &SI)\n{\n    tide::op_profile::Scope profile(tide::op_profile::KvBuild);\n    int64_t N'),
        ('            return cache.slice(2,0,maxendidx,1).index({SI});',
         '            tide::op_profile::Scope profile(tide::op_profile::KvGather);\n            return cache.slice(2,0,maxendidx,1).index({SI});'),
        ('    // qs:: SI.size(0)',
         '    tide::op_profile::Scope profile(tide::op_profile::Attention);\n    // qs:: SI.size(0)')])
    change('src/CortexNet.cpp', [
        ('            Tensor local_emit_signal = bcn.nets.anymodules[i].forward(',
         '            Tensor local_emit_signal;\n            { tide::op_profile::Scope profile(tide::op_profile::Emit);\n            local_emit_signal = bcn.nets.anymodules[i].forward('),
        ('            VTensor local_emit_signals = local_emit_signal.split',
         '            }\n            VTensor local_emit_signals = local_emit_signal.split'),
        ('        res[j]->x = this->norm.anymodules[j].forward',
         '        tide::op_profile::Scope profile(tide::op_profile::Norm);\n        res[j]->x = this->norm.anymodules[j].forward')])
    return modified
