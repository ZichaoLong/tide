#!/usr/bin/env python3
"""Review raw independent benchmark repeats without inventing missing samples."""
import argparse
import json
from pathlib import Path
import statistics
from durable_records import write_json,replace_text


def summarize(suite):
    groups={}
    for run in suite['runs']:
        config=run['config'];key=(config['id'],config['graph'],config['body_nodes'],run['variant'])
        groups.setdefault(key,[]).append(run)
    rows=[]
    for (identity,family,n,variant),runs in groups.items():
        modes=sorted({m.split('/')[1] for r in runs for m in r['metrics'] if m.endswith('/seconds')})
        if not modes:modes=['unmeasured']
        for mode in modes:
            values=[r['metrics'][f'perf/{mode}/seconds'] for r in runs if r['status']=='completed' and f'perf/{mode}/seconds' in r['metrics']]
            row=dict(id=identity,graph=family,body_nodes=n,variant=variant,mode=mode,
                     completed_repeats=len(values),failed_repeats=sum(r['status']!='completed' for r in runs),
                     independent_seconds=values,dispersion_qualified=len(values)>=3,
                     failures=[dict(directory=r['directory'],error=r['error'],model_constructed=r['model_constructed'])
                               for r in runs if r['status']!='completed'])
            if values:
                row.update(median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values),
                           coefficient_of_variation=statistics.stdev(values)/statistics.mean(values) if len(values)>1 else None,
                           median_ms_per_sample_position=statistics.median(r['metrics'][f'perf/{mode}/ms_per_sample_position']
                              for r in runs if r['status']=='completed'))
            rows.append(row)
    return dict(schema='tide-foundation-summary-v1',source=suite['source'],tier=suite['tier'],
                evaluation_state=suite['state'],rows=rows,bounded_stops=suite['bounded_stops'],
                scope='Measured durations, not an automatic speedup/default recommendation; fewer than three samples do not establish gain')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(json.loads(a.suite.read_text()));write_json(a.output,result)
    lines=['# Frozen foundation benchmark observations','',
           '| ID / graph | variant | mode | repeats / failures | median seconds | min–max | CV |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for r in result['rows']:
        timing=f"{r['median_seconds']:.5g} | {r['min_seconds']:.5g}–{r['max_seconds']:.5g} | {r['coefficient_of_variation']:.3g}" if r.get('coefficient_of_variation') is not None else f"{r.get('median_seconds','unmeasured')} | — | —"
        lines.append(f"| {r['id']} / {r['graph']} / N{r['body_nodes']} | {r['variant']} | {r['mode']} | {r['completed_repeats']} / {r['failed_repeats']} | {timing} |")
    lines+=['',result['scope'],'']
    replace_text(a.output.with_suffix('.md'),'\n'.join(lines))


if __name__=='__main__':main()
