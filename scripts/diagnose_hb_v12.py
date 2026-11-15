#!/usr/bin/env python3
"""Use the existing fixed-window observer with v12 identities."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_capacity_v11 import main
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','config','table','data','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--lens-cli',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--recording-group',required=True)
    a=p.parse_args();a.index=0;main(a)
