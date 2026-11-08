#!/usr/bin/env python3
"""Freeze previously unused test articles; preserve all training/validation rows."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from tide_lab.records import Metrics,atomic_json,digest,now,project_to_lens
from tide_lab.language_data import choose_documents,shifted_block,text_hash


def select_confirmation(documents,old_lock,seed=20261109,count=16):
    used={d['id'] for d in old_lock['selected_documents']['test']}
    pool=[d for d in documents if d['split']=='test' and d['id'] not in used]
    selected=choose_documents(pool,count,seed)
    if len(selected)!=count or len({d['id'] for d in selected})!=count:
        raise ValueError('insufficient distinct unused test documents')
    return selected


def prepare(a):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from tide_lab.growth import verify_checkpoint
    from tide_lab.train import source_identity
    torch.set_num_threads(2)
    started=time.monotonic();old=json.loads((a.parent/'data-lock.json').read_text())
    source=json.loads((a.documents/'documents-lock.json').read_text())
    if digest(a.documents/'documents.jsonl')!=source['documents_sha256']:
        raise ValueError('source documents changed')
    if digest(a.parent/'tokenizer-a.json')!=old['tokenizer_a_sha256']:
        raise ValueError('A tokenizer changed')
    hashes=verify_checkpoint(a.asset_path)
    if hashes['tokenizer.json']!=old['tokenizer_b_sha256']:
        raise ValueError('B tokenizer changed')
    documents=[json.loads(line) for line in (a.documents/'documents.jsonl').read_text().splitlines()]
    selected=select_confirmation(documents,old)
    a.output.mkdir(parents=True,exist_ok=False);out=a.output/'tokens';out.mkdir()
    run_id='tl-lm0-confirmation-v4'
    atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',
        study_id='TL-LM0',group='TL-LM0',run_id=run_id,name=run_id,created_at=now(),
        config=dict(source=source_identity(),parent_data_lock_sha256=digest(a.parent/'data-lock.json'),
                    selection_seed=20261109,documents=16)))
    atomic_json(a.output/'state.json',dict(state='running'));metrics=Metrics(a.output,run_id)
    # Publish selection before tokenization, cache building, or any outcome.
    selection=dict(schema='tide-confirmation-selection-v1',created_at=now(),seed=20261109,
        ids=[d['id'] for d in selected],excluded=[d['id'] for d in old['selected_documents']['test']],
        rule='first 16 unused official-test articles by sha256(seed:id); first two 128-target chunks')
    atomic_json(a.output/'selection.json',selection)
    tokenizer=Tokenizer.from_file(str(a.parent/'tokenizer-a.json'))
    qwen=AutoTokenizer.from_pretrained(a.asset_path,local_files_only=True,trust_remote_code=False)
    lock=copy.deepcopy(old)
    lock['selected_documents']['test']=[{k:d[k] for k in ('id','sha256','utf8_bytes','title')} for d in selected]
    lock['confirmation']=dict(parent_data_lock_sha256=digest(a.parent/'data-lock.json'),
        selection_sha256=digest(a.output/'selection.json'),selection_seed=20261109,
        training_validation='tensor values, row order and identifiers exactly preserved',
        old_test='observed development evidence, excluded from confirmation')
    shutil.copyfile(a.parent/'tokenizer-a.json',out/'tokenizer-a.json')
    for route,encode in [('a',lambda text:tokenizer.encode(text).ids),
                         ('b',lambda text:qwen.encode(text,add_special_tokens=False))]:
        info=old['routes'][route];parent=a.parent/info['file']
        if digest(parent)!=info['sha256']:raise ValueError('parent token artifact changed')
        tensors=torch.load(parent,map_location='cpu',weights_only=False)
        blocks=[];length=old['sequence_length']
        for doc in selected:
            tokens=encode(doc['text'])+[info['eos']]
            for offset in range(0,min(len(tokens),2*length),length):
                x,y=shifted_block(tokens[offset:offset+length],length=length,bos=info['bos'],pad=info['pad'])
                blocks.append(dict(x=x,y=y,document_id=doc['id'],id=f'{doc["id"]}:{offset}'))
        tensors['test']=dict(input_ids=torch.tensor([b['x'] for b in blocks]),
            labels=torch.tensor([b['y'] for b in blocks]),document_ids=[b['document_id'] for b in blocks],
            block_ids=[b['id'] for b in blocks])
        torch.save(tensors,out/info['file'])
        reloaded=torch.load(out/info['file'],map_location='cpu',weights_only=False)
        original=torch.load(parent,map_location='cpu',weights_only=False)
        for split in ('train','validation'):
            for key,value in original[split].items():
                actual=reloaded[split][key]
                if not (torch.equal(value,actual) if isinstance(value,torch.Tensor) else value==actual):
                    raise AssertionError('training/validation preservation failed')
        lock['routes'][route]['sha256']=digest(out/info['file'])
        lock['routes'][route]['splits']['test']=dict(available_blocks=len(blocks),blocks=len(blocks),
            valid_tokens=int((tensors['test']['labels']!=-100).sum()),document_ids=sorted({b['document_id'] for b in blocks}),
            block_ids_sha256=text_hash(json.dumps([b['id'] for b in blocks])))
        metrics.log(0 if route=='a' else 1,{route+'/confirmation_tokens':lock['routes'][route]['splits']['test']['valid_tokens']})
    atomic_json(out/'data-lock.json',lock)
    for path in out.iterdir():path.chmod(0o444)
    (a.output/'selection.json').chmod(0o444)
    result=dict(state='completed',data_lock_sha256=digest(out/'data-lock.json'),
        elapsed_seconds=time.monotonic()-started,training_validation_exact=True,selection=selection)
    atomic_json(a.output/'result.json',result);atomic_json(a.output/'state.json',result)
    metrics.close();atomic_json(a.output/'completed.json',dict(producer_finished=True,state='completed'))
    result['lens_projection_completed']=project_to_lens(a.output,str(a.recording_cli))
    atomic_json(a.output/'result.json',result)
    print(json.dumps(result));return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','documents','output','asset-path','recording-cli'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:return prepare(a)
    except BaseException as error:
        traceback.print_exc()
        if a.output.is_dir() and (a.output/'manifest.json').exists():
            result=dict(state='failed',error=repr(error),finished_at=now(),traceback=traceback.format_exc())
            atomic_json(a.output/'state.json',result);atomic_json(a.output/'result.json',result)
            atomic_json(a.output/'completed.json',dict(producer_finished=True,state='failed'))
        return 1


if __name__=='__main__':raise SystemExit(main())
