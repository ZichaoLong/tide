"""Auditable metadata-only instrumentation of the a10fdb1 CROSSBATCH CPU test.

Changes are applied only to a new prepared copy; anchors must be unique.
Original tensor expressions, clearing, selection and scheduling stay intact.
"""
from pathlib import Path
import shutil


def instrument(source, root):
    cpp = Path(source) / 'Connectome/cpp'
    modified = []

    def change(name, edits):
        path = cpp / name
        text = path.read_text()
        for before, after in edits:
            if text.count(before) != 1:
                raise ValueError('accounting anchor mismatch: ' + name + ': ' + before)
            text = text.replace(before, after)
        path.write_text(text)
        modified.append(path)

    change('src/AccumulateLocal.cpp', [
        ('Tensor AL::Attention::forward_via_multi_batch', '#include "tide/operator_work.h"\nTensor AL::Attention::forward_via_multi_batch'),
        ('VTensor qkvs = this->c_attn(x)', 'tide::work::linear(tide::work::QkvCalls, x.size(0), this->D, 3*this->D);\n    VTensor qkvs = this->c_attn(x)'),
        ('return this->c_proj(output);', 'tide::work::linear(tide::work::OutCalls, N, this->D, this->D);\n    return this->c_proj(output);')])
    change('src/BatchHidden.cpp', [
        ('Tensor BatchPtrKVHidden::attention(', '#include "tide/operator_work.h"\nTensor BatchPtrKVHidden::attention('),
        ('    allscores = (allscores+', '''    if (tide::work::enabled()) {
        int64_t valid = 0;
        auto ends = localendindicesT.accessor<int64_t,1>();
        auto queries = Icpu.accessor<int64_t,1>();
        for (int64_t j = 0; j < N; ++j) valid += ends[j]*queries[j];
        tide::work::attention(D, n_head, valid, SIN*allscores.size(2));
    }
    allscores = (allscores+''')])
    change('src/CortexNet.cpp', [
        ('#include "CortexNet.h"', '#include "CortexNet.h"\n#include "tide/operator_work.h"'),
        ('            Tensor local_emit_signal =', '''            tide::work::emit(ptr->x.size(0), bcn.config.emitD, k1-k0);
            Tensor local_emit_signal ='''),
        ('        /* 这一步还需将 chs[j] decay*/', '''        if (tide::work::enabled() && bchali[j])
            tide::work::add(tide::work::BodyCandidates, bchali[j]->get_sampleids().size(0));
        /* 这一步还需将 chs[j] decay*/'''),
        ('        if (!res[j]) return;', '''        if (!res[j]) return;
        if (tide::work::enabled()) tide::work::add(tide::work::BodySelected, res[j]->x.size(0));''')])
    change('include/CortexNet.h', [
        ('struct Pronounce :', '#include "tide/operator_work.h"\nstruct Pronounce :'),
        ('{return this->lm_head->forward', '''{tide::work::linear(tide::work::HeadCalls, bi_ptr->get_sampleids().size(0), D, vocab_size);
     return this->lm_head->forward''')])
    change('test/test-cortexnet.cpp', [
        ('int main()', '#include "lh_accounting.h"\nint main()'),
        ('    int64_t batch_size =', '    lh_accounting::initialize();\n    int64_t batch_size ='),
        ('Tensor ids = at::randint(0, ionet.config.vocab_size, {batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(ionet.iodevice));',
         'Tensor ids = at::remainder(at::arange(batch_size, torch::TensorOptions().dtype(torch::kInt64).device(ionet.iodevice))*3+t*7, ionet.config.vocab_size);'),
        ('        Tensor logits = ionet.think', '''        tide::work::reset(lh_accounting::counting());
        tide::op_profile::reset(lh_accounting::profiling());
        auto begin = std::chrono::steady_clock::now();
        Tensor logits = ionet.think'''),
        ('        cout << "t: " << t << endl;', '''        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        cout << "t: " << t << endl;
        lh_accounting::finish(t, logits, ionet, iacts, oacts, elapsed);''')])
    # Kept inside the prepared tree and explicitly hashed in its manifest.
    copies = {'cpp/include/tide/operator_work.h': 'include/tide/operator_work.h',
              'cpp/include/tide/operator_profile.h': 'include/tide/operator_profile.h',
              'cpp/include/portable_torch/threads.hpp': 'include/portable_torch/threads.hpp',
              'cpp/src/threads.cpp': 'src/tide_threads.cpp',
              'cpp/lh_original/accounting.h': 'include/lh_accounting.h'}
    added = []
    for src, dst in copies.items():
        path = cpp / dst
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / src, path)
        added.append(path)
    from lh_profile_instrument import instrument as profile_instrument
    modified.extend(profile_instrument(cpp))
    return modified, added


def parse_work(text, steps):
    import json
    events = [json.loads(line[5:]) for line in text.splitlines() if line.startswith('WORK ')]
    if [e['step'] for e in events] != list(range(steps)):
        raise ValueError('accounting token inventory mismatch')
    return events
