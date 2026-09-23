#!/usr/bin/env python3
"""Export clean source with verifiable provenance; no Git or original tree needed at runtime."""
import argparse
from pathlib import Path
import shutil
import subprocess
from durable_records import write_json
from source_identity import source_state,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    commit,dirty=source_state(root)
    if dirty:parser.error('export requires clean committed source')
    files=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in sorted(set(files)-{''}):
        src=root/name;dest=out/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        hashes[name]=digest(dest)
    write_json(out/'source-export.json',dict(schema='tide-foundation-source-v1',commit=commit,files_sha256=hashes))
    assert source_state(out)==(commit,'')
    print(out)


if __name__=='__main__':main()
