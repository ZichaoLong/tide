"""Named finite witnesses retain strict failure and cannot hide new mismatches."""
import pytest
import torch
from dense_checks import Checks
from tidegraph.comparison import NumericMismatch, NonFiniteMismatch


def test_named_witness_stays_xfail_and_other_coordinates_stay_strict(monkeypatch):
    monkeypatch.delenv("TIDE_DENSE_NUMERIC_REPORT", raising=False)
    expected = {"evidence": torch.zeros(2), "other": torch.zeros(2)}
    witness = dict(expected, evidence=torch.tensor([1.4e-6, 0.]))
    checker = Checks("named", known_numeric={"root[evidence]": 1.6e-6})
    checker(witness, expected)
    with pytest.raises(pytest.xfail.Exception):
        checker.finish()
    for changed in (dict(expected, evidence=torch.tensor([2e-6, 0.])),
                    dict(witness, other=torch.tensor([2e-6, 0.]))):
        with pytest.raises(NumericMismatch):
            Checks("new", known_numeric={"root[evidence]": 1.6e-6})(changed, expected)
    with pytest.raises(NonFiniteMismatch):
        checker(dict(expected, evidence=torch.tensor([float('nan'), 0.])), expected)
