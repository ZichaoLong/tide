"""The original timer is a text protocol: reject missing/reordered completions."""
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('benchmark_lh_original', SCRIPTS / 'benchmark_lh_original.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_original_timer_keeps_cold_step_and_ignores_nested_timers():
    values, parameters = module.parse_log('构造 ionet, 参数量 = 123456\n'
        'Start Think ...\nIntra forward took 99 ms \nThink took 1500 ms \nt: 0\n'
        'Start Think ...\nThink took 800 ms \nt: 1\n')
    assert values == [1500, 800]
    assert parameters == 123456


@pytest.mark.parametrize('text', ['t: 0\n', 'Think took 50 ms \nt: 1\n',
                                 'Think took 50 ms \nThink took 60 ms \nt: 0\n'])
def test_original_timer_rejects_unpaired_or_reordered_tokens(text):
    with pytest.raises(ValueError):
        module.parse_log(text)
