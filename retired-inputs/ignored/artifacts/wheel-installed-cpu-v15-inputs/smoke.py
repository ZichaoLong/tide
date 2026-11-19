"""Run only the installed public Tide package from an isolated outside cwd."""
import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import fields,is_dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import runpy
import sys
import sysconfig
import tempfile
import zipfile


def atomic(path,value):
    path=Path(path)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.smoke.')
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def tensors(value):
    import torch
    if isinstance(value,torch.Tensor):yield value
    elif is_dataclass(value):
        for field in fields(value):yield from tensors(getattr(value,field.name))
    elif isinstance(value,dict):
        for key in sorted(value,key=str):yield from tensors(value[key])
    elif isinstance(value,(tuple,list)):
        for item in value:yield from tensors(item)


def fingerprint(value):
    result=hashlib.sha256()
    for tensor in tensors(value):
        result.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        result.update(tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--example',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--source',required=True)
    args=parser.parse_args()
    import torch
    import tide
    from tide import ModelConfig,TopologyConfig,TideModel
    from tide.precision import ArithmeticAudit,precision_context,precision_policy
    torch.set_num_threads(4)
    purelib=Path(sysconfig.get_paths()['purelib']).resolve()
    assert Path.cwd()==Path('/') and 'PYTHONPATH' not in os.environ
    assert sys.flags.isolated and sys.flags.no_user_site
    assert Path(tide.__file__).resolve().is_relative_to(purelib)
    assert Path(tide.__file__).resolve()==Path(importlib.metadata.distribution('tide-instances').locate_file('tide/__init__.py')).resolve()
    with zipfile.ZipFile(args.wheel) as archive:
        hashes={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                if name.startswith('tide/') and name.endswith('.py')}
    assert all(hashlib.sha256((purelib/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    record=dict(source=args.source,state='running',scope='installed-public-API mechanism smoke; no expanded precision or performance qualification',
                cwd=str(Path.cwd()),python=sys.version,executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,
                purelib=str(purelib),sys_path=sys.path,pythonpath=os.environ.get('PYTHONPATH'),isolated=bool(sys.flags.isolated),
                tide_file=tide.__file__,tide_version=importlib.metadata.version('tide-instances'),
                wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),installed_files_match_wheel=True,
                installed_python_file_count=len(hashes),torch=str(torch.__version__),torch_file=torch.__file__,
                backend_autoload=os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD'),cpu_affinity=sorted(os.sched_getaffinity(0)),
                torch_threads=torch.get_num_threads(),example_sha256=hashlib.sha256(args.example.read_bytes()).hexdigest(),rows=[])
    atomic(args.output,record)
    try:
        for precision in ('fp32','bf16'):
            torch.manual_seed(271118)
            config=ModelConfig(topology=TopologyConfig(n=8,layers=4),d=64,memory='K')
            model=TideModel(config).eval();x=torch.randn(2,8,64)
            parameter_hash,input_hash=fingerprint(model.state_dict()),fingerprint(x)
            audit=ArithmeticAudit()
            with torch.inference_mode(),precision_context('cpu',precision),audit:
                inferred=model(x)
                readout=inferred.readout()
            assert all(bool(torch.isfinite(t).all()) for t in tensors(inferred) if t.is_floating_point())
            arithmetic=audit.report()
            expected='cpu:bfloat16' if precision=='bf16' else 'cpu:float32'
            assert any(expected in row['outputs'] for row in arithmetic)
            row=dict(precision=precision,state='running',policy=precision_policy('cpu',precision),
                     config=config.to_dict(),batch=2,length=8,inference_seed=271118,
                     parameter_sha256=parameter_hash,input_sha256=input_hash,inference_finite=True,
                     inference_readout_shape=list(readout.shape),inference_state_position=inferred.state.position,
                     inference_arithmetic=arithmetic)
            record['rows'].append(row);atomic(args.output,record)
            checkpoint=args.output.parent/(precision+'-checkpoint.pt')
            argv=[str(args.example),'--device','cpu','--precision',precision,'--memory','K','--steps','3',
                  '--threads','4','--checkpoint',str(checkpoint)]
            row['example_command']=[sys.executable,'-I',*argv]
            capture=io.StringIO();previous=sys.argv
            try:
                sys.argv=argv
                with redirect_stdout(capture):runpy.run_path(str(args.example),run_name='__main__')
            finally:
                sys.argv=previous
                row['example_stdout']=capture.getvalue()
            observations=[ast.literal_eval(line) for line in capture.getvalue().splitlines() if line.strip()]
            assert len(observations)==4 and [x['step'] for x in observations[:3]]==[0,1,2]
            restored=observations[-1]
            assert restored['restored_optimizer'] and restored['continuation']=='exact'
            assert restored['next_optimizer_update'] in {'within_threshold','finite_deviation'}
            row.update(state='passed',training_seed=17,training_updates=observations[:3],restore=restored,
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       checkpoint_bytes=checkpoint.stat().st_size)
            atomic(args.output,record)
        origins={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
                 if (name=='tide' or name.startswith('tide.')) and getattr(module,'__file__',None)}
        assert all(Path(path).is_relative_to(purelib) for path in origins.values())
        record.update(state='passed',loaded_tide_modules=origins)
    except BaseException as error:
        record.update(state='failed',error=repr(error))
        raise
    finally:
        atomic(args.output,record)


if __name__=='__main__':main()
