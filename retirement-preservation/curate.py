"""Retain unique source bytes; bound historical numerical/profiler JSON records."""
import base64
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import os

root=Path(__file__).resolve().parent
inventory=json.loads((root/'inventory.json').read_text())
assert json.loads((root/'preserve-status.json').read_text())['state']=='passed'
src=root/'small-records.jsonl.gz';out=root/'curated-records.jsonl.gz'
counts=Counter();summarized=set()

def summary(value,depth=0):
    if isinstance(value,dict):
        if depth>=6:return {'retired_mapping_keys':len(value)}
        keys=list(value)
        r={k:summary(value[k],depth+1) for k in keys[:64]}
        if len(keys)>64:r['_retired_additional_keys']=len(keys)-64
        return r
    if isinstance(value,list):
        if len(value)>24 or depth>=6:
            return {'retired_list_length':len(value),'first_item_type':type(value[0]).__name__ if value else None}
        return [summary(v,depth+1) for v in value]
    if isinstance(value,str) and len(value)>4096:return value[:2048]+' [retired remainder]'
    return value

with gzip.open(src,'rt') as source,gzip.open(out,'wt',compresslevel=9) as target:
    for line in source:
        row=json.loads(line)
        if row.get('retained')=='bundle_bytes':
            data=base64.b64decode(row['data_base64'])
            suffix=Path(row['path']).suffix
            if suffix in ('.json','.jsonl') and len(data)>32768:
                del row['data_base64'];row['original_sha256']=hashlib.sha256(data).hexdigest()
                row['retained']='summary_only';summarized.add(row['git_blob'])
                try:row['summary']=summary(json.loads(data))
                except (ValueError,UnicodeDecodeError):row['summary']={'unparsed_tail':data[-2048:].decode(errors='replace')}
                counts['json_summarized']+=1;counts['raw_json_bytes_retired']+=len(data)
            else:counts['exact_bundle_bytes']+=len(data);counts['exact_unique_files']+=1
        elif row.get('retained')=='bundle_duplicate' and row['git_blob'] in summarized:
            row['retained']='summary_duplicate'
        counts['records']+=1
        target.write(json.dumps(row,ensure_ascii=False)+'\n')
inventory['raw_collection_identity']=inventory['preservation_files']['small-records.jsonl.gz']
os.replace(out,src)
inventory['curation']=dict(counts)
inventory['preservation_files']['small-records.jsonl.gz']={
    'bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
(root/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'curation':dict(counts),'bundle':inventory['preservation_files']},indent=2))
