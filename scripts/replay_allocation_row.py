#!/usr/bin/env python3
"""Replay a captured allocation through the installed public scalar API."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tidegraph.weighted import allocate
from tide_lab.records import atomic_json,digest


def replay(record,dtype,temperature_factor=1.):
    record=record.get('captured',record.get('failure',record))
    fields={k:torch.tensor(v['values'],dtype=dtype).reshape(v['shape'])
            for k,v in record['tensors'].items()}
    if record['details']['regularizer']!='ent' or float(record['details']['alpha'])!=1.:
        raise ValueError('this reproducer is scoped to captured Shannon rows')
    try:
        answer=allocate(fields['scores'],fields['caps'],fields['lower'],fields['upper'],
            fields['temperature']*temperature_factor,regularizer='ent',ent_alpha=1.,score_shift=fields.get('shift'))
        return dict(state='completed',controls=answer.controls.tolist(),multiplier=float(answer.multiplier),
                    residual=float(answer.residual),constraint=answer.constraint)
    except ArithmeticError as error:
        result=dict(state='failed',error=repr(error))
        if callable(getattr(error,'record',None)):result['failure']=error.record()
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--dtype',choices=['float32','float64'],required=True)
    p.add_argument('--temperature-factor',type=float,default=1.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise FileExistsError(a.output)
    result=replay(json.loads(a.record.read_text()),getattr(torch,a.dtype),a.temperature_factor)
    result.update(record_sha256=digest(a.record),dtype=a.dtype,temperature_factor=a.temperature_factor,
                  scope='single captured row; no full-model, gradient, or recovery claim')
    atomic_json(a.output,result);print(json.dumps(result))
    raise SystemExit(0 if result['state']=='completed' else 1)
