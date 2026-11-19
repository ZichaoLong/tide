"""Post-retirement isolated-wheel public CPU training and actual resume smoke."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = '97c52960358e6d5414fc987dc3861b1e85d997a9'
sys.path.insert(0, str(ROOT/'scripts'))
from job import atomic

scratch = HERE/'smoke-scratch'
scratch.mkdir(exist_ok=False)
source = scratch/'build-copy'; source.mkdir()
archive = subprocess.check_output(['git','archive',REVISION,'pyproject.toml','src','examples','configs/recommended'], cwd=ROOT)
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    tar.extractall(source, filter='data')
environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
environment.pop('PYTHONPATH',None)
python = ROOT/'.venv-instances/bin/python'
def run(argv, name, cwd='/'):
    p = subprocess.run([str(x) for x in argv], cwd=cwd, env=environment, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (HERE/name).write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f'{name}: exit {p.returncode}')
    return p.stdout
run([python,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',scratch/'wheel',source], 'smoke-build.log')
wheel, = (scratch/'wheel').glob('*.whl')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
with zipfile.ZipFile(wheel) as z:
    members = [n for n in z.namelist() if n.startswith('tide/') and n.endswith('.py')]
    hashes = {n:hashlib.sha256(z.read(n)).hexdigest() for n in members}
    expected = {'tide/'+p.name:sha(p) for p in (source/'src/tide').glob('*.py')}
    assert hashes == expected
venv = scratch/'venv'
run([python,'-m','venv','--system-site-packages','--without-pip',venv], 'smoke-venv.log')
installed_python = venv/'bin/python'
run([python,'-m','pip','--python',installed_python,'install','--ignore-installed','--no-deps','--no-index',wheel], 'smoke-install.log')
worker = scratch/'worker.py'
worker.write_text('''import hashlib,json,runpy,sys
from pathlib import Path
sys.argv=sys.argv[1:]
try:
    runpy.run_path(sys.argv[0],run_name="__main__")
except SystemExit as error:
    if error.code not in (None,0): raise
import tide
root=Path(tide.__file__).resolve().parent
assert root.is_relative_to(Path(sys.prefix).resolve())
assert sys.flags.isolated
assert all(Path(m.__file__).resolve().is_relative_to(root) for n,m in sys.modules.items() if n=="tide" or n.startswith("tide."))
print(json.dumps(dict(kind="installed_identity",package_root=str(root),isolated=sys.flags.isolated,
    files={"tide/"+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")},vendor_imported="torch_npu" in sys.modules)))
''')
entry = source/'examples/train.py'; checkpoint = scratch/'checkpoint.pt'
train = run([installed_python,'-I',worker,entry,'--threads','1','--steps','1','--checkpoint',checkpoint], 'smoke-train.log')
resume = run([installed_python,'-I',worker,entry,'--threads','1','--steps','1','--resume',checkpoint], 'smoke-resume.log')
def records(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith('{')]
tr, rr = records(train), records(resume)
for record in (tr[-1],rr[-1]):
    assert record['kind']=='installed_identity' and record['files']==hashes and not record['vendor_imported']
check, = [r for r in tr if r['kind']=='restore_check']
assert check['checkpoint_data']=='exact' and check['next_forward']=='exact' and check['next_optimizer_update']=='within_threshold'
step, = [r for r in rr if r['kind']=='step']
assert step['optimizer_updated'] and step['state_position']>step['state_before_position']
atomic(HERE/'installed-smoke.json',dict(state='passed',source=REVISION,wheel_sha256=sha(wheel),
    wheel_files=hashes,device='cpu',precision='fp32',threads=1,affinity=sorted(os.sched_getaffinity(0)),
    training_records=tr,fresh_process_records=rr,scope='isolated public entry and local recovery smoke; no broad numerical or task-quality qualification'))
print(json.dumps(dict(state='passed',source=REVISION,package_files=len(hashes))))
