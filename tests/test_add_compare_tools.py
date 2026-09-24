"""Add profile ownership counts, original timer independence and prepared-source options."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools/cpu_compare'), str(ROOT/'scripts')]
from compare_build import prepare, build_commands, command
from compare_config import parameters
from compare_metrics import Collector, summarize


def test_add_parameter_count_and_no_attention_padding():
    a = SimpleNamespace(width=2048, vocab=50304, memory='add')
    assert parameters(a) == 9_468_020_899
    a.memory = 'attention'
    assert parameters(a) == 17_269_426_339
    events = [dict(step=i, metrics={'perf/ms_per_sample_token': 2.,
              'op/qkv_rows': 0., 'op/executed_score_elements': 0.,
              'op/valid_score_elements': 0., 'op/executed_matmul_flops': 20.}) for i in range(3)]
    assert summarize(events, 4, 1)['attention_padding_ratio'] is None


def test_outer_timer_does_not_need_raii_output(tmp_path):
    a = SimpleNamespace(batch=4, warmup=1, steps=3, work_count=0, lh_timer='outer')
    collector = Collector('lh', a, tmp_path, 'outer', 23)
    collector.line('构造 ionet, 参数量 = 23')
    for i, seconds in enumerate((.1, .123456, .2)):
        collector.line(f't: {i}')
        collector.line('WORK '+json.dumps(dict(step=i, metrics={'perf/token_seconds': seconds})))
    result = collector.finish()
    assert result['raw_ms'] == pytest.approx([30.864, 50.])
    assert result['mean_ms_per_sample_token'] == pytest.approx(40.432)
    with pytest.raises(ValueError, match='order'):
        collector.line('t: 5')


def test_add_preparation_changes_owned_copy_only(tmp_path):
    kit = tmp_path/'kit'; source = kit/'sources/lh'
    (source/'test').mkdir(parents=True)
    (kit/'graph-data').mkdir(); (kit/'vendor').mkdir()
    model = {n:dict(emitD=2048, receiveD=2048) for n in ('input_', 'output_', 'iobridge_', 'oibridge_')}
    for n in ('input_', 'output_'):
        model[n]['chals'] = [dict(chal='attention', confluence='allsoftmax', decay_rate=.01)]*2
    model.update(vocab_size=50304, pronounce=dict(chal='attention', confluence='allsoftmax'))
    (source/'test/cfg.json').write_text(json.dumps(model))
    original = 'int64_t batch_size = 512;\nfor (int t=0; t<12; ++t) {}\n'
    (source/'test/test-cortexnet.cpp').write_text(original)
    (source/'CMakeLists.txt').write_text('target_compile_definitions(Connectome PUBLIC DISABLE_CUDA ENABLE_RAIITIMER)\n')
    out = tmp_path/'run'; out.mkdir()
    a = SimpleNamespace(kit_dir=str(kit), memory='add', lh_timer='outer', mode='grad-forward',
                        width=16, batch=4, steps=6, vocab=257, jobs=2)
    copied = prepare('lh', a, dict(lh_batch=512, lh_steps=12), out)
    cfg = json.loads((copied/'test/cfg.json').read_text())
    assert cfg['pronounce']['chal'] == 'add'
    assert all(c['chal']=='add' for n in ('input_', 'output_') for c in cfg[n]['chals'])
    assert all(cfg[n]['emitD']==16 for n in ('input_', 'output_', 'iobridge_', 'oibridge_'))
    assert 'ENABLE_RAIITIMER' not in (copied/'CMakeLists.txt').read_text()
    assert 'ENABLE_RAIITIMER' in (source/'CMakeLists.txt').read_text()
    assert (source/'test/test-cortexnet.cpp').read_text() == original
    _, _, binary = build_commands('lh', a, copied, '/not-used')
    assert binary.name == 'test-cortexnet-grad'
