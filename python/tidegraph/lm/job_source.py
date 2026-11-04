"""Fixed-source admission for initial and delayed durable submissions."""
import subprocess


def source_identity(source):
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    if dirty:
        raise ValueError('qualification source must be a clean fixed checkout')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
