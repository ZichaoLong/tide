"""LM result projection, reusing the project's offline HTML and atomic records."""
import csv
import io
import json
from pathlib import Path
import statistics
from .catalog import model
from ..experiment.records import read_json, replace_text

COLUMNS = ('name','model','parameters','platform','mode','logical_batch','positions','output_tokens',
           'dtype','client','placement','devices','threads','execution_status','numerical','discrete',
           'checkpoint','median_seconds','output_tokens_per_s','decode_tokens_per_s','target_tokens_per_s',
           'host_peak_bytes','device_peak_bytes','record')


def load(root):
    rows=[]
    for path in sorted(Path(root).rglob('result.json')):
        value=read_json(path)
        if value.get('schema')!='tide-lm-result-v1':
            continue
        c=value['configuration'];t=c['task'];e=c['execution'];v=value['validation']
        measurements=value.get('measurements',[])
        def median(key):
            values=[x[key] for x in measurements if key in x]
            return statistics.median(values) if values else None
        rows.append(dict(name=c['name'],model=c['model'],parameters=model(c['model'])['weight_parameters'],
            platform=e['backend'],mode=t['mode'],logical_batch=t['batch'],positions=t['input_tokens'],
            output_tokens=t['output_tokens'] if t['mode']=='inference' else None,dtype=e['dtype'],client=e['engine'],
            placement=e['placement'],devices=e['devices'],threads=e['threads'],execution_status=value['execution_status'],
            numerical=v['numeric'],discrete=v['discrete'],checkpoint=v['checkpoint'],
            median_seconds=median('total_seconds'),output_tokens_per_s=median('output_tokens_per_s'),
            decode_tokens_per_s=median('decode_tokens_per_s'),target_tokens_per_s=median('target_tokens_per_s'),
            host_peak_bytes=value.get('host_peak_rss_bytes'),device_peak_bytes=value.get('device_peak'),
            record=str(path),detail=value))
    return rows


def render(rows, format='table'):
    if format=='json':
        return json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if format=='html':
        from ..experiment.html_report import render_html
        filters=tuple((k,k) for k in ('model','platform','mode','logical_batch','dtype','client','placement','execution_status','checkpoint'))
        groups=(('Model',(('model','model'),('parameters','parameters'))),
                ('Work',(('mode','task'),('logical_batch','batch'),('positions','input tokens'),('output_tokens','output cap'))),
                ('Execution',(('platform','platform'),('client','engine'),('dtype','dtype'),('devices','devices'),('threads','threads'))),
                ('Facts',(('execution_status','execution'),('numerical','numeric'),('discrete','tokens'),('checkpoint','resume'))),
                ('Cost',(('median_seconds','seconds'),('output_tokens_per_s','output tokens/s'),('decode_tokens_per_s','decode tokens/s'),('target_tokens_per_s','train target tokens/s'))))
        return render_html(rows,filters=filters,groups=groups,title='External dense / MoE language models')
    columns=COLUMNS if format=='csv' else ('name','execution_status','median_seconds','output_tokens_per_s','target_tokens_per_s','checkpoint')
    if format not in {'table','csv'}:
        raise ValueError('format: table/csv/html/json required')
    stream=io.StringIO();writer=csv.writer(stream,delimiter=',' if format=='csv' else '\t')
    writer.writerow(columns)
    for row in rows:
        writer.writerow([json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in columns])
    return stream.getvalue()


def save_all(root):
    rows=load(root)
    for format in ('csv','html'):
        replace_text(Path(root)/('report.'+format),render(rows,format))
