"""Read explicit diagnostic attachments without rerunning a workload.

Costs, smaller profiles, independent checks and terminal jobs have different
identities. Preserve that separation; never derive qualification from timings.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from source_identity import digest
from collect_fc_comparison import cpu_profile, npu_profile, numeric_section


def census(record):
    """Native counters include registered private graphs; do not add outer twice."""
    out={'scope':record.get('scope')}
    py=record.get('python')
    if py:
        out['python']={k:v for k,v in py.items() if k in ('total','groups','scope')}
    native=record.get('native')
    if native:
        out['native']={k:v for k,v in native.items()
                       if k.count('/')==1 or (k.startswith('graphs/') and k.endswith('/unique_nodes'))}
        out['native_scope']='Registered outer/private graphs deduplicated within native; saved tensors explicitly instrumented only.'
    return out


def profile(path, d, exports=None):
    if d.get('schema')!='tide-dense-query-cost-v1':
        raise ValueError('profile result needs an explicit schema adapter')
    result={k:d[k] for k in ('profile','profile_state','instrumented_step',
                             'instrumented_stages','stage_scope','census_scope') if k in d}
    result['identity']={k:d[k] for k in ('foundation_source','foundation_dirty','resolved',
        'native_build','mode','baseline','query_execution','trace_records','precision',
        'initial_parameters_sha256','profile_parameters_sha256','census_parameters_sha256') if k in d}
    result['packing_totals']={k:v for k,v in d.get('instrumented_packing',{}).items()
                              if v and k.count('/')<=3}
    if d.get('profile_state')!='completed':
        result['unavailable']='Profiler did not record a completed terminal state.'
    elif d['profile']=='cpu':
        result['cpu']=cpu_profile(path.parent)
    elif d['profile']=='npu':
        # Export relocation is explicit and does not alter measurement identity.
        result['npu']=npu_profile(path.parent,d,exports=exports)
    elif d['profile'] not in ('stages','none'):
        raise ValueError('unknown requested profile type')
    if isinstance(d.get('autograd_census'),str):
        cp=Path(d['autograd_census']);cp=cp if cp.is_absolute() else path.parent/cp
        result['autograd']=census(json.loads(cp.read_text())['autograd'])
        result['census_sha256']=digest(cp)
    result['scope']='Diagnostic interval/shape only. No throughput from instrumented timing or extrapolation to a different B/L.'
    return result


def attachments(items,directory):
    results=[];names=set()
    for item in items:
        if item['name'] in names:raise ValueError('duplicate diagnostic name')
        names.add(item['name'])
        path=Path(item['path']);path=path if path.is_absolute() else directory/path
        row=dict(name=item['name'],kind=item['kind'],path=str(path.resolve()),sha256=digest(path))
        if 'source' in item:row['declared_source']=item['source']
        if item['kind']=='junit':
            root=ET.parse(path)
            row['suites']=[dict(s.attrib) for s in root.iter('testsuite')]
            row['nonpasses']=[dict(case=c.attrib,results=[dict(kind=x.tag,attributes=x.attrib,text=x.text)
                for x in c if x.tag in ('failure','error','skipped')]) for c in root.iter('testcase')
                if any(x.tag in ('failure','error','skipped') for x in c)]
        else:
            d=json.loads(path.read_text())
            if item['kind']=='profile':
                export=item.get('exports')
                if export is not None:
                    export=Path(export);export=export if export.is_absolute() else directory/export
                row.update(profile(path,d,export))
            elif item['kind']=='projection-check':
                if d['schema']!='tide-fc-projection-comparison-v1':raise ValueError('unknown projection-check schema')
                row.update({k:d[k] for k in ('execution','request','scope','tolerance','variants')})
                row['comparisons']={n:{p:numeric_section(x) for p,x in v.items()}
                                    for n,v in d['comparisons'].items()}
            elif item['kind']=='job':
                if d.get('state') not in ('passed','failed','cancelled'):raise ValueError('job must be terminal')
                row['job']=d
            else:raise ValueError('unknown diagnostic kind')
        results.append(row)
    return results
