#!/usr/bin/env python3
"""Check progress-only C++ changes against the original binary on fixed tiny inputs.

The caller owns the compute lock. No new performance qualification is implied.
Compare exact tokens/logits/full state and each binary's fresh-process resume.
"""
import argparse
from pathlib import Path
from tidegraph.lm.matrix import case
from tidegraph.lm.config import resolve
from tidegraph.lm.llama_adapter import invoke,material,progress_events
from tidegraph.experiment.records import read_json,write_json,digest,now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--previous-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=read_json(args.previous_build);new=read_json(root/'engines/llama/ready.json')
    result=dict(state='running',started=now(),before=old,after=new,models={})
    write_json(out/'status.json',result)
    try:
        for key in ('qwen15','olmoe'):
            c=case(key,'cpu',2,16,kind='progress-regression',output_tokens=4,warmup=0,
                   checkpoint=True,attention='eager')['config']
            c['execution']['engine']='llama_cpp';c=resolve(c)['request']
            conversion=read_json(root/'engines/llama'/key/'conversion.json')
            outputs={}
            for label,build in (('before',old),('after',new)):
                target=out/key/label;target.mkdir(parents=True)
                material(c,target)
                recorded=dict(build=build,conversion=conversion)
                invoke(c,root/'models',target,'run',recorded)
                invoke(c,root/'models',target,'restore',recorded)
                outputs[label]={f:digest(target/f) for f in ('tokens.bin','logits.bin','state.bin',
                    'expected-state.bin','expected-logits.bin','restored-state.bin','restored-logits.bin')}
            checks={name:outputs['before'][name]==outputs['after'][name] for name in outputs['before']}
            resumes={label:all(h['expected-'+field+'.bin']==h['restored-'+field+'.bin'] for field in ('state','logits'))
                     for label,h in outputs.items()}
            events=progress_events(out/key/'after/run.log')
            complete=bool(events) and events[-1]['phase']=='run' and events[-1]['stage']=='complete'
            result['models'][key]=dict(configuration=c,hashes=outputs,exact_before_after=checks,
                                      exact_resumes=resumes,progress_complete=complete,progress=events)
            write_json(out/'status.json',result)
            if not all(checks.values()) or not all(resumes.values()) or not complete:
                raise RuntimeError('progress regression differs; do not run performance successors')
        result.update(state='passed',finished=now())
        write_json(out/'status.json',result)
    except BaseException as error:
        result.update(state='failed',finished=now(),error=str(error)[:1500]);write_json(out/'status.json',result);raise


if __name__=='__main__':main()
