#!/usr/bin/env python3
"""One CPU admission chain; its result never certifies NPU execution."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.language_driver import StageRunner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','data','output','asset-path','recording-cli'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args();a.route='a';a.cpu_only=True;plan=json.loads(a.plan.read_text())
    runner=StageRunner(a,plan,{'cpu':850},850)
    try:
        admitted=True
        for kind in plan['a_candidates']:
            _,result=runner.stage('cpu-check-'+kind,'checks','cpu',180,['--kind',kind,'--length','3'])
            admitted=admitted and result['state']=='completed' and result.get('admitted_for_finite_exploration',False)
        if admitted:
            runner.stage('cpu-calibrate-tide-dense-L32-B8','calibrate','cpu',180,
                         ['--kind','tide-dense','--length','32','--batch','8'])
        runner.state['independent_cpu_admission']=admitted
        runner.finish('completed' if admitted else 'failed')
        return 0 if admitted else 1
    except BaseException as error:
        runner.finish('failed',repr(error));raise


if __name__=='__main__':raise SystemExit(main())
