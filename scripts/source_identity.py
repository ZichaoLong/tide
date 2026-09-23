"""Identity of a Git checkout or an explicitly exported immutable source bundle."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_state(root):
    manifest=root/'source-export.json'
    if manifest.exists():
        data=json.loads(manifest.read_text())
        if data.get('schema')!='tide-foundation-source-v1':
            raise ValueError('unsupported source export schema')
        for name,expected in data['files_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('invalid exported source path')
            if digest(root/name)!=expected:
                raise ValueError('exported source changed: '+name)
        return data['commit'],''
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    return commit,dirty
