"""Render a small, offline research graph; no model/runtime dependencies."""
import argparse
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'goal', 'baseline', 'experiment', 'result', 'evidence'}
RUN_STATUSES = {'planned', 'queued', 'running', 'completed', 'failed', 'unchecked'}
STATUSES = {'target', 'unselected', 'documented', 'awaiting', 'scoped'} | RUN_STATUSES
RELATIONS = {'decomposes', 'precondition', 'variant', 'produces', 'comparison',
             'transfer', 'informs', 'evaluates', 'supports', 'challenges'}


def relative_reference(value, location):
    """Repository-relative links never contain instance URLs or credentials."""
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError(f'{location}: expected repository-relative path')
    parsed = urlparse(value)
    path = PurePosixPath(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'{location}: expected repository-relative path')


def public_source(value, location):
    parsed = urlparse(value)
    host = parsed.hostname or ''
    local = host == 'localhost' or host.endswith(('.localhost', '.local', '.internal'))
    try:
        local = local or not ipaddress.ip_address(host).is_global
    except ValueError:
        pass
    if parsed.scheme != 'https' or not host or parsed.username or parsed.password or parsed.query or local:
        raise ValueError(f'{location}: expected public HTTPS source without private instance details')


def validate(data):
    if data.get('schema') != 'tide-research-map-prototype-v1':
        raise ValueError('schema: unsupported research map')
    nodes, studies, runs = {}, set(), {}
    for node in data['nodes']:
        key = node['id']
        if not key or key in nodes:
            raise ValueError(f'nodes.{key}: duplicate/empty id')
        nodes[key] = node
        if node['kind'] not in KINDS or node['status'] not in STATUSES:
            raise ValueError(f'nodes.{key}: unknown kind/status')
        for field in ('title', 'summary', 'question', 'decision', 'budget'):
            if not isinstance(node.get(field), str) or not node[field].strip():
                raise ValueError(f'nodes.{key}.{field}: expected text')
        for field, limit in [('x', data['width'] - 250), ('y', data['height'] - 118)]:
            if type(node[field]) is not int or not 0 <= node[field] <= limit:
                raise ValueError(f'nodes.{key}.{field}: outside canvas')
        if node['status'] == 'scoped' and (not node.get('sources') or not isinstance(node.get('scope'), str) or not node['scope'].strip()):
            raise ValueError(f'nodes.{key}: actual evidence needs sources and scope')
        if node['kind'] == 'result' and node['status'] not in {'awaiting', 'scoped'}:
            raise ValueError(f'nodes.{key}: result must be awaiting or scoped')
        if node['kind'] == 'result' and node['status'] == 'scoped':
            for field in ('comparisons', 'counterevidence', 'unresolved', 'transfer_conditions'):
                if not isinstance(node.get(field), str) or not node[field].strip():
                    raise ValueError(f'nodes.{key}: reviewed result needs {field}')
        if node.get('study_id'):
            if node['study_id'] in studies:
                raise ValueError(f'nodes.{key}: duplicate study_id')
            studies.add(node['study_id'])
        for source in node.get('sources', []):
            public_source(source['url'], f'nodes.{key}.sources')
        for reference in node.get('references', []):
            relative_reference(reference['path'], f'nodes.{key}.references')
        for run in node.get('runs', []):
            if not isinstance(run.get('id'), str) or not run['id'].strip() or run.get('status') not in RUN_STATUSES:
                raise ValueError(f'nodes.{key}.runs: expected run ID and explicit lifecycle status')
            if run['id'] in runs and runs[run['id']] != run:
                raise ValueError(f'nodes.{key}.runs: inconsistent shared run identity')
            runs[run['id']] = run
            for field in ('config', 'record'):
                relative_reference(run[field], f'nodes.{key}.runs.{field}')
            if run.get('lens_record'):
                relative_reference(run['lens_record'], f'nodes.{key}.runs.lens_record')
    seen = set()
    for edge in data['edges']:
        if edge['id'] in seen:
            raise ValueError('edges: duplicate id')
        seen.add(edge['id'])
        if 'via_y' in edge and (type(edge['via_y']) is not int or not 0 <= edge['via_y'] < data['height']):
            raise ValueError('edge via_y must be a finite in-canvas integer')
        if edge['source'] not in nodes or edge['target'] not in nodes:
            raise ValueError(f'edges.{edge["id"]}: unknown endpoint')
        if edge['kind'] not in RELATIONS or edge['status'] not in {'planned', 'documented'}:
            raise ValueError(f'edges.{edge["id"]}: unknown kind/status')
        if edge['kind'] in {'supports', 'challenges'} or (edge['kind'] == 'transfer' and edge['status'] == 'documented'):
            origin = nodes[edge['source']]
            if origin['kind'] not in {'result', 'evidence'} or origin['status'] != 'scoped':
                raise ValueError(f'edges.{edge["id"]}: support/refutation/reviewed transfer requires actual scoped evidence')
    # Only the prerequisite relation is a DAG. Comparisons/transfer are not ancestry.
    active, done = set(), set()
    children = {key: [] for key in nodes}
    for edge in data['edges']:
        if edge['kind'] == 'precondition':
            children[edge['source']].append(edge['target'])
    def visit(key):
        if key in active:
            raise ValueError('precondition edges: cycle')
        if key in done:
            return
        active.add(key)
        for target in children[key]:
            visit(target)
        active.remove(key)
        done.add(key)
    for key in nodes:
        visit(key)


def render(root=ROOT):
    data = json.loads((root/'research/map.json').read_text())
    validate(data)
    encoded = json.dumps(data, ensure_ascii=False).replace('<', r'\u003c').replace('&', r'\u0026')
    template = (root/'web/map.template.html').read_text()
    return template.replace('__STYLE__', (root/'web/map.css').read_text()).replace(
        '__SCRIPT__', (root/'web/map.js').read_text()).replace('__DATA__', encoded)


def publish(path, content):
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check generated HTML is current; do not write')
    args = parser.parse_args()
    content = render()
    path = ROOT/'web/index.html'
    if args.check:
        if not path.is_file() or path.read_text() != content:
            raise SystemExit('web/index.html is stale; run python scripts/render_map.py')
        print('Graph references, evidence boundaries and generated HTML: checked')
    else:
        publish(path, content)
        print('Generated web/index.html (offline, no model execution)')


if __name__ == '__main__':
    main()
