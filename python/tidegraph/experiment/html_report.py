"""Offline result comparison; no runtime imports, network or external assets."""
import html
import json


FILTERS = (
    ("platform", "Platform"), ("graph", "Graph"), ("topology", "Topology"),
    ("logical_batch", "Batch"), ("dtype", "Payload dtype"), ("scoring_dtype", "Scoring dtype"),
    ("client", "Client"), ("placement", "Placement"), ("mode", "Task"),
    ("execution_status", "Execution"), ("qualification", "Strict checks"),
    ("return_mode", "Return"), ("packing", "Packing"), ("return_packing", "Return packing"),
    ("communication", "Communication"), ("schedule", "Schedule"), ("threads", "ATen threads"),
    ("workers", "Workers"),
)


def _text(value):
    if value is None:
        return ""
    return str(value)


def _escape(value):
    return html.escape(_text(value), quote=True)


def _lines(row, fields):
    def value(key):
        item = row.get(key)
        return "unmeasured" if item is None else f"{item:.8g}" if isinstance(item, float) else item
    return "<br>".join(f'<span class="label">{_escape(label)}:</span> {_escape(value(key))}'
                       for key, label in fields)


# Filters only inspect encoded attributes; untrusted records never enter JS code.
FILTER_SCRIPT = r"""
function matchesRecord(record, filters, minimum, maximum) {
  for (const [key, value] of Object.entries(filters)) {
    if (value !== '' && record[key] !== value) return false;
  }
  if (minimum !== '' || maximum !== '') {
    if (!/^\d+$/.test(record.parameters) || !/^\d*$/.test(minimum) || !/^\d*$/.test(maximum)) return false;
    const count = BigInt(record.parameters);
    if (minimum !== '' && count < BigInt(minimum)) return false;
    if (maximum !== '' && count > BigInt(maximum)) return false;
  }
  return true;
}
function applyFilters() {
  const filters = {};
  document.querySelectorAll('#filters select').forEach(x => { filters[x.name] = x.value; });
  const minimum = document.getElementById('params-min').value;
  const maximum = document.getElementById('params-max').value;
  const valid = /^\d*$/.test(minimum) && /^\d*$/.test(maximum);
  document.getElementById('filter-error').textContent = valid ? '' : 'Parameter bounds require nonnegative decimal integers.';
  if (!valid) return;
  let count = 0;
  document.querySelectorAll('#records tr').forEach(row => {
    const record = {parameters: row.getAttribute('data-parameters')};
    Object.keys(filters).forEach(key => { record[key] = row.getAttribute('data-' + key); });
    row.hidden = !matchesRecord(record, filters, minimum, maximum);
    if (!row.hidden) count++;
  });
  document.getElementById('count').textContent = count + ' matching records';
}
document.getElementById('filters').addEventListener('input', applyFilters);
document.getElementById('filters').addEventListener('change', applyFilters);
document.getElementById('filters').addEventListener('reset', () => setTimeout(applyFilters, 0));
applyFilters();
"""


def render_html(rows):
    controls = []
    for key, label in FILTERS:
        values = sorted({_text(row.get(key)) for row in rows} - {""})
        options = '<option value="">All</option>' + "".join(
            f'<option value="{_escape(value)}">{_escape(value)}</option>' for value in values)
        controls.append(f'<label>{label}<select name="{key}">{options}</select></label>')
    controls += [
        '<label>Minimum parameters<input id="params-min" inputmode="numeric" pattern="[0-9]*" placeholder="e.g. 1000000000"></label>',
        '<label>Maximum parameters<input id="params-max" inputmode="numeric" pattern="[0-9]*"></label>',
        '<button type="reset">Reset filters</button>',
    ]
    groups = (
        ("Model", (("topology", "topology"), ("graph", "graph"), ("parameters", "parameters"))),
        ("Workload", (("logical_batch", "B"), ("positions", "positions"), ("span", "span"),
                      ("mode", "task"), ("dtype", "payload"), ("scoring_dtype", "scoring"),
                      ("schedule", "schedule"))),
        ("Execution", (("platform", "platform"), ("client", "client"), ("placement", "placement"),
                       ("devices", "devices"), ("communication", "communication"),
                       ("threads", "ATen threads"), ("workers", "workers"))),
        ("Status", (("execution_status", "execution"), ("qualification", "strict"),
                    ("exploratory", "exploratory"))),
        ("Independent checks", (("numerical", "numeric"), ("discrete", "discrete"),
                                ("complete", "complete"))),
        ("Continuation", (("checkpoint", "checkpoint"), ("exact_resume", "exact resume"))),
        ("Window cost", (("median_seconds", "median seconds"), ("samples_per_second", "samples/s"),
                         ("return_mode", "return"), ("packing", "packing"),
                         ("return_packing", "return packing"), ("throughput_qualified", "qualified"))),
    )
    body = []
    for row in rows:
        attrs = " ".join(f'data-{key}="{_escape(row.get(key))}"' for key in
                         [key for key, _ in FILTERS] + ["parameters"])
        cells = "".join("<td>"+_lines(row, fields)+"</td>" for _, fields in groups)
        detail = html.escape(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))
        cells += '<td><details><summary>Details</summary><pre>'+detail+'</pre></details></td>'
        body.append(f'<tr {attrs}>{cells}</tr>')
    headings = "".join(f'<th scope="col">{title}</th>' for title, _ in groups)
    return ('''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tidegraph results</title>
<style>body{font:14px system-ui,sans-serif;margin:1.5em;color:#17212b;background:#fafbfc}
h1{margin-bottom:.3em}p{max-width:90em;line-height:1.5}form{display:flex;flex-wrap:wrap;gap:.7em;
padding:1em;background:#edf1f5;border-radius:6px}label{display:flex;flex-direction:column;gap:.3em}
select,input,button{font:inherit;padding:.3em;border:1px solid #a6b3c0;border-radius:3px;background:white}
button{align-self:end;cursor:pointer}.table-scroll{overflow:auto;max-height:75vh}
table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d6dde4;padding:.6em;
text-align:left;vertical-align:top}th{position:sticky;top:0;background:#e8edf3;z-index:1}
td{min-width:8em;line-height:1.5}.label{color:#526578}pre{white-space:pre-wrap;min-width:24em;
max-width:70em;max-height:35em;overflow:auto;overflow-wrap:anywhere}summary{cursor:pointer}
tr[hidden]{display:none}</style></head><body><h1>Tidegraph results</h1>
<p>Execution, independent checks and exact resume are separate. Unmeasured is never passed.
Compare timings only for matching models, inputs, work, precision, return contracts and resources.
Exploratory costs are not qualified throughput. Expand Details for complete configuration,
work counts, memory, failures and source identities.</p>
<noscript>All records are shown. Enable JavaScript for offline filtering, or use tidegraph results filters.</noscript>
<form id="filters" onsubmit="return false">''' + "".join(controls) + '</form>'
            f'<p id="filter-error" role="alert"></p><p id="count" role="status">{len(rows)} records</p><div class="table-scroll"><table><thead><tr>'
            + headings + '<th scope="col">Full record</th></tr></thead><tbody id="records">'
            + "".join(body) + '</tbody></table></div><script type="text/javascript">'
            + FILTER_SCRIPT + '</script></body></html>\n')
