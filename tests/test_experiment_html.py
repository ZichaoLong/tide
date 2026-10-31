"""Offline browsing must preserve records and never hide failures implicitly."""
from html.parser import HTMLParser
import json
import shutil
import subprocess

import pytest

from tidegraph.experiment.html_report import FILTER_SCRIPT, render_html
from tidegraph.experiment.report import select


class Page(HTMLParser):
    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.rows, self.scripts, self.details = [], [], []
        self.active = None
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        if tag == "tr" and attrs:
            self.rows.append(dict(attrs))
        if tag in {"script", "pre"}:
            self.active = self.scripts if tag == "script" else self.details
            self.active.append("")

    def handle_data(self, text):
        if self.active is not None:
            self.active[-1] += text

    def handle_endtag(self, tag):
        if tag in {"script", "pre"}:
            self.active = None


def test_offline_page_preserves_full_records_and_escapes_filter_attributes():
    attack = 'cuda" onclick="alert(1)"><script>bad()</script>&'
    rows = [dict(platform=attack, parameters=123, logical_batch=512,
                 execution_status="completed", qualification="strict-failure",
                 numerical="failed", discrete="passed", checkpoint="passed",
                 details=dict(error=attack, complete_configuration={"a": 2}))]
    page = Page(render_html(rows))
    assert len(page.scripts) == 1 and page.scripts[0] == FILTER_SCRIPT
    assert page.rows[0]["data-platform"] == attack
    assert "onclick" not in page.rows[0]
    assert [json.loads(text) for text in page.details] == rows
    assert page.rows[0]["data-qualification"] == "strict-failure"


def test_empty_page_and_missing_measurements_are_explicit():
    assert '0 records' in render_html([])
    assert 'unmeasured' in render_html([dict(median_seconds=None)])


def test_browser_predicate_matches_public_filters_for_boundaries_and_failures():
    node = shutil.which("node")
    if not node:
        pytest.skip("optional offline JavaScript execution check requires Node")
    rows = [dict(platform="cpu", parameters=1, logical_batch=128, qualification="strict-failure"),
            dict(platform="cuda", parameters=1000000000, logical_batch=512, qualification="strict-checks-passed"),
            dict(platform="npu", parameters=None, logical_batch=512, qualification="not-established"),
            dict(platform="cpu", parameters=9007199254740993, logical_batch=1, qualification="not-established")]
    requests = [{}, dict(platform="cpu"), dict(logical_batch=512), dict(params_min=1),
                dict(params_max=1), dict(params_min=1000000000, params_max=1000000000),
                dict(qualification="strict-failure"), dict(platform="npu", params_min=0),
                dict(params_min=9007199254740993, params_max=9007199254740993),
                dict(params_max=9007199254740992)]
    expected = [[rows.index(row) for row in select(rows, **request)] for request in requests]
    data = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    script = "const document={getElementById:()=>({value:'',addEventListener(){}}),querySelectorAll:()=>[]};\n"
    requests = [{key: str(value) for key, value in request.items()} for request in requests]
    script += FILTER_SCRIPT + "\nconst rows=" + json.dumps(data) + ";const requests=" + json.dumps(requests) + ";\n"
    script += """const actual=requests.map(request=>{
      const filters=Object.fromEntries(Object.entries(request).filter(([k])=>!k.startsWith('params_'))
        .map(([k,v])=>[k,String(v)]));
      return rows.flatMap((row,i)=>matchesRecord(row,filters,
        request.params_min===undefined?'':String(request.params_min),
        request.params_max===undefined?'':String(request.params_max))?[i]:[]);
    });console.log(JSON.stringify(actual));"""
    checked = subprocess.run([node], input=script, text=True, capture_output=True, check=True, timeout=10)
    assert json.loads(checked.stdout) == expected
