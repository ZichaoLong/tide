import hashlib,json,pathlib,sys,os,tempfile
job=pathlib.Path(sys.argv[1]);expected=json.loads((job/'source-manifest.json').read_text())['files'];source=job/'frozen-source'
actual={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
result=dict(state='passed' if expected==actual else 'failed',files=len(actual),changed=sorted(k for k in expected.keys()|actual.keys() if expected.get(k)!=actual.get(k)))
fd,stage=tempfile.mkstemp(dir=job,prefix='.source-verify.')
with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(stage,job/'source-unchanged.json');fd=os.open(job,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
if expected!=actual:raise SystemExit(1)
