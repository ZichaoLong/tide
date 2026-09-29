"""CANN can split a full-size operator export into consecutive CSV slices."""
import csv
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "ascend_profile_summary", Path(__file__).parents[1]/"scripts/summarize_ascend_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def write(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def operators(path, rows):
    write(path, ["Task Type", "OP Type", "Task Duration(us)", "Input Data Types"], rows)


def api(root):
    write(root/"api_statistic_20260929.csv", ["Level", "API Name", "Time(us)", "Count"],
          [["runtime", "Launch", "9", "3"]])


def test_slices_preserve_counts_durations_and_every_input_hash(tmp_path):
    api(tmp_path)
    operators(tmp_path/"op_summary_slice_0_20260929.csv", [["AI_CORE", "MatMul", "2", "FLOAT"]])
    operators(tmp_path/"op_summary_slice_1_20260929.csv", [["AI_CPU", "Sort", "3", "INT64"],
                                                         ["AI_CORE", "MatMul", "5", "FLOAT"]])
    result = profile.summarize(tmp_path)
    assert result["operators"] == 3 and result["summed_task_us"] == 10
    assert result["engines"]["AI_CORE"] == dict(count=2, summed_task_us=7, summed_task_fraction=.7)
    assert result["engines"]["AI_CPU"]["count"] == 1
    assert set(result["inputs"]) == {p.name for p in tmp_path.glob("*.csv")}


def test_unsliced_export_retains_same_summary(tmp_path):
    api(tmp_path)
    operators(tmp_path/"op_summary_20260929.csv", [["AI_CORE", "MatMul", "4", "FLOAT"]])
    result = profile.summarize(tmp_path)
    assert result["operators"] == 1 and result["summed_task_us"] == 4


@pytest.mark.parametrize("names", [
    [],
    ["op_summary_slice_1_20260929.csv"],
    ["op_summary_slice_0_20260929.csv", "op_summary_slice_2_20260929.csv"],
    ["op_summary_slice_0_20260929.csv", "op_summary_slice_1_20260930.csv"],
    ["op_summary_slice_0_20260929.csv", "op_summary_slice_00_20260929.csv"],
    ["op_summary_20260929.csv", "op_summary_slice_0_20260929.csv"],
    ["op_summary_20260929.csv", "op_summary_20260930.csv"],
])
def test_rejects_incomplete_or_ambiguous_exports(tmp_path, names):
    for name in names:
        (tmp_path/name).touch()
    with pytest.raises(ValueError):
        profile.operator_files(tmp_path)


def test_slices_sort_by_numeric_index(tmp_path):
    for i in range(12):
        (tmp_path/f"op_summary_slice_{i}_20260929.csv").touch()
    assert [p.name.split("_")[3] for p in profile.operator_files(tmp_path)] == list(map(str, range(12)))
