"""Portable log reports must distinguish completed work from failed processes."""
import importlib.util
from pathlib import Path
import pytest

FILE = Path(__file__).resolve().parents[1]/'tools/lh_repro/summarize.py'
SPEC = importlib.util.spec_from_file_location('lh_repro_summary', FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG = dict(batch=4, steps=4, expected_parameters=23, seed_policy='original unseeded')
LOG = '构造 ionet, 参数量 = 23\n' + ''.join(
    f'Think took {time} ms \nt: {index}\n' for index, time in enumerate([100,120,140,160]))


def test_original_timer_units_and_cold_step_are_preserved():
    report = MODULE.summarize(LOG, CONFIG, 0)
    assert report['status'] == 'completed'
    assert report['windows']['all']['mean_ms_per_sample_token'] == 32.5
    assert report['windows']['all']['tokens_per_second'] == pytest.approx(1000/32.5)
    assert report['raw_think_ms_per_batch'] == [100,120,140,160]
    assert 'late-80-99' not in report['windows']


@pytest.mark.parametrize('text,code', [(LOG, 1), (LOG+'Think took 20 ms\n', 0),
    (LOG.replace('t: 3\n', ''), 0), (LOG.replace('参数量 = 23', '参数量 = 24'), 0)])
def test_incomplete_or_failed_process_never_passes(text, code):
    assert MODULE.summarize(text, CONFIG, code)['status'] == 'failed-or-incomplete'


def test_timer_token_mispairing_rejected():
    with pytest.raises(ValueError, match='sequence'):
        MODULE.summarize(LOG.replace('t: 2', 't: 8'), CONFIG, 0)
