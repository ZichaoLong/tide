"""A matching core/backend cannot qualify an executable from stale client code."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from source_identity import digest
from verify_resident_target import check_consumer_sources


@pytest.fixture
def consumer(tmp_path):
    directory = tmp_path/'tools/online_bench'
    directory.mkdir(parents=True)
    for name, text in {'CMakeLists.txt': 'add_executable(client main.cpp)',
                       'main.cpp': 'int main() {return 0;}',
                       'options.h': '#pragma once',
                       'resident.py': 'VERSION = 1'}.items():
        (directory/name).write_text(text)
    record = {'consumer_sources': {p.name: digest(p) for p in directory.iterdir()}}
    return tmp_path, directory, record


@pytest.mark.parametrize('change', ['edited-cpp', 'edited-header', 'added', 'removed', 'missing-record'])
def test_rejects_stale_consumer_even_with_unchanged_core(consumer, change):
    root, directory, record = consumer
    if change == 'edited-cpp':
        (directory/'main.cpp').write_text('int main() {return 1;}')
    elif change == 'edited-header':
        (directory/'options.h').write_text('#define DIFFERENT_BEHAVIOR 1')
    elif change == 'added':
        (directory/'extra.h').write_text('#pragma once')
    elif change == 'removed':
        (directory/'options.h').unlink()
    else:
        record.pop('consumer_sources')
    with pytest.raises(ValueError, match='compiled sources differ'):
        check_consumer_sources(root, record, 'online_bench')


def test_fixed_checkout_python_changes_do_not_stale_cpp_binary(consumer):
    root, directory, record = consumer
    check_consumer_sources(root, record, 'online_bench')
    (directory/'resident.py').write_text('VERSION = 2')
    check_consumer_sources(root, record, 'online_bench')
