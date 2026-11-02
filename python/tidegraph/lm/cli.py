"""External LM configuration, assets, runs, independent checks and offline reports."""
import argparse
import json
from pathlib import Path
from .catalog import catalog
from ..experiment.records import read_json, replace_text


def main(argv=None):
    parser=argparse.ArgumentParser(prog='tidegraph lm',description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('catalog',help='fixed model revisions and asset sizes; no model allocation')
    commands.add_parser('matrix',help='print the finite authorized manifest and unqueued engine points')
    fetch=commands.add_parser('fetch',help='one download attempt; explicit rerun resumes partial files')
    fetch.add_argument('model',choices=tuple(catalog()))
    fetch.add_argument('--root',required=True);fetch.add_argument('--project-root',required=True)
    fetch.add_argument('--project-gib',type=int,default=80);fetch.add_argument('--account-gib',type=int,default=300)
    plan=commands.add_parser('plan',help='validate, expand and budget JSON without allocating a model')
    plan.add_argument('config')
    for name in ('run','check','training-check'):
        p=commands.add_parser(name,help='one foreground run' if name=='run' else 'small independent CPU FP32 schedule check')
        p.add_argument('config');p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
        p.add_argument('--case-wrapper',action='store_true',help=argparse.SUPPRESS)
    p=commands.add_parser('resume',help='verify exact next step in a fresh process using a saved checkpoint')
    p.add_argument('run_dir');p.add_argument('--assets',required=True)
    p=commands.add_parser('results',help='read existing results; never starts model work')
    p.add_argument('root');p.add_argument('--format',choices=('table','csv','html','json'),default='table')
    p.add_argument('--output');p.add_argument('--model');p.add_argument('--backend');p.add_argument('--state')
    p=commands.add_parser('queue',help='bounded matrix worker; launch in a durable service')
    p.add_argument('--root',required=True);p.add_argument('--backend',choices=('cpu','npu'),required=True)
    p.add_argument('--npu-helper');p.add_argument('--hours',type=float,default=23.5)
    p.add_argument('--phase',choices=('first','second'),default='first')
    p.add_argument('--attempt',default='a',help='unique immutable attempt suffix; failures are never overwritten')
    p=commands.add_parser('_worker',help=argparse.SUPPRESS)
    p.add_argument('case');p.add_argument('--root',required=True);p.add_argument('--case-dir',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='catalog':
            result={key:dict(id=value['id'],revision=value['revision'],stage=value['stage'],
                            bytes=sum(f['bytes'] for f in value['files'])) for key,value in catalog().items()}
        elif args.command=='fetch':
            from .assets import fetch as run
            result=run(args.model,args.root,args.project_root,project_gib=args.project_gib,account_gib=args.account_gib)
        elif args.command=='plan':
            from .config import resolve
            result=resolve(args.config)
        elif args.command=='matrix':
            from .matrix import manifest
            result=manifest()
        elif args.command in {'run','check','training-check'}:
            from .config import resolve
            config=read_json(args.config)
            if args.case_wrapper:config=config['config']
            config=resolve(config)['request']
            if args.command=='run':
                from .engine import run
            elif args.command=='check':
                from .verification import check as run
            else:
                from .training_check import check as run
            result=run(config,args.assets,args.output)
        elif args.command=='resume':
            from .checkpoint import resume
            result=resume(args.run_dir,args.assets)
            print(json.dumps(result,indent=2));return 0 if result['state']=='passed' else 1
        elif args.command=='results':
            from .report import load,render
            rows=load(args.root)
            rows=[x for x in rows if (not args.model or x['model']==args.model)
                  and (not args.backend or x['platform']==args.backend)
                  and (not args.state or x['execution_status']==args.state)]
            text=render(rows,args.format)
            if args.output:replace_text(Path(args.output),text)
            else:print(text,end='')
            return 0
        elif args.command=='_worker':
            from .queue import worker
            return worker(args.case,args.root,args.case_dir)
        else:
            from .queue import execute
            if not 0<args.hours<=24:raise ValueError('--hours must be in (0,24]')
            result=execute(args.root,args.backend,npu_helper=args.npu_helper,hours=args.hours,phase=args.phase,attempt=args.attempt)
        print(json.dumps(result,indent=2,ensure_ascii=False));return 0
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1,f'{type(error).__name__}: {error}\n')
