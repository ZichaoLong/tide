import pytest
from tidegraph.compare import equivalent
from foundation_training import trajectory


CASES = [(family, implementation) for family in ('pdg','dag','settle')
         for implementation in ('native-serial','native-parallel','native-packed','specialized','native-specialized')
         if not (family == 'settle' and implementation == 'native-specialized')]


@pytest.mark.parametrize('family,implementation', CASES)
@pytest.mark.parametrize('memory', ['ssm','attention'])
@pytest.mark.parametrize('kind', ['sgd','momentum','adamw'])
def test_six_classes_multistep_truncation_owners_and_optimizer(dtype, family, implementation, memory, kind):
    expected = trajectory(dtype, family, 'reference', memory, kind)
    actual = trajectory(dtype, family, implementation, memory, kind, native_optimizer=implementation.startswith('native'))
    equivalent(expected, actual)
