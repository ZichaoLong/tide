"""Document-bounded natural language data; no model or backend imports at import time."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .records import atomic_json, digest


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_documents(rows, split):
    """WikiText level-one titles delimit documents; keep original text bytes."""
    documents, current, title, start = [], [], None, 0
    def flush(end):
        if not current or not ''.join(current).strip():
            return
        if title is None:
            raise ValueError('nonempty WikiText preamble before first article title')
        text = ''.join(current)
        documents.append(dict(id=f'{split}-{len(documents):04d}-{text_hash(text)[:12]}',
            split=split, title=title, start_row=start, end_row=end, text=text,
            sha256=text_hash(text), normalized_sha256=text_hash(' '.join(text.split())),
            utf8_bytes=len(text.encode('utf-8'))))
    for index, row in enumerate(rows):
        heading = re.fullmatch(r'= ([^=].*?[^=]|[^=]) =', row.strip())
        if heading:
            flush(index)
            current, title, start = [], heading.group(1), index
        current.append(row)
    flush(len(rows))
    if not documents:
        raise ValueError('no WikiText articles parsed')
    return documents


def deduplicate(documents):
    """Test is never tuned: exact heldout overlap fails; exclude train overlaps."""
    kept, removed = {}, []
    for split in ('validation', 'test', 'train'):
        seen = set()
        kept[split] = []
        heldout = {d['normalized_sha256'] for s in ('validation', 'test')
                   for d in kept.get(s, [])}
        for doc in documents[split]:
            h = doc['normalized_sha256']
            reason = 'within_split_duplicate' if h in seen else None
            if split == 'test' and h in heldout:
                raise ValueError('validation/test article overlap; explicit split redesign required')
            if split == 'train' and h in heldout:
                reason = 'train_heldout_exact_article_overlap'
            seen.add(h)
            if reason:
                removed.append(dict(id=doc['id'], reason=reason))
            else:
                kept[split].append(doc)
    return kept, removed


def shifted_block(tokens, *, length, bos, pad):
    """One chunk predicts every target once; input never contains that target early."""
    if not tokens or len(tokens) > length:
        raise ValueError('expected a nonempty, bounded target chunk')
    valid = len(tokens)
    return ([bos, *tokens[:-1], *([pad]*(length-valid))],
            [*tokens, *([-100]*(length-valid))])


def choose_documents(docs, count, seed):
    return sorted(docs, key=lambda d: text_hash(f'{seed}:{d["id"]}'))[:count]


def extract(source, output):
    import pyarrow
    import pyarrow.parquet as pq
    source, output = Path(source), Path(output)
    lock = json.loads((source/'source.json').read_text())
    for row in lock['files']:
        if digest(source/row['filename']) != row['sha256']:
            raise ValueError('source identity mismatch: '+row['filename'])
    docs, source_rows = {}, {}
    for split in ('train', 'validation', 'test'):
        rows = pq.read_table(source/f'{split}-00000-of-00001.parquet')['text'].to_pylist()
        source_rows[split] = len(rows)
        docs[split] = parse_documents(rows, split)
    counts = {s: len(rows) for s, rows in docs.items()}
    docs, removed = deduplicate(docs)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'documents.jsonl').open('x') as stream:
        for split in ('train', 'validation', 'test'):
            for row in docs[split]:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    metadata = dict(schema='tide-language-documents-v1', source=lock,
        source_rows=source_rows, original_article_counts=counts,
        article_counts={s:len(rows) for s,rows in docs.items()}, removed=removed,
        documents_sha256=digest(output/'documents.jsonl'), pyarrow=pyarrow.__version__,
        boundary='exact level-one WikiText titles; original row bytes joined',
        duplicate_scope='exact whitespace-normalized whole articles; near-duplicate and pretraining contamination unknown')
    atomic_json(output/'documents-lock.json',metadata)
    return metadata


def tokenize(documents_dir, output, asset, *, length=128, vocab=4096):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import AutoTokenizer
    from .growth import verify_checkpoint
    documents_dir, output = Path(documents_dir), Path(output)
    source_lock = json.loads((documents_dir/'documents-lock.json').read_text())
    if digest(documents_dir/'documents.jsonl') != source_lock['documents_sha256']:
        raise ValueError('document bytes changed')
    docs = {s:[] for s in ('train','validation','test')}
    for line in (documents_dir/'documents.jsonl').read_text().splitlines():
        row=json.loads(line); docs[row['split']].append(row)
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator((d['text'] for d in docs['train']),
        trainers.BpeTrainer(vocab_size=vocab,min_frequency=2,
            special_tokens=['<pad>','<bos>','<eos>','<unk>'],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    if tokenizer.get_vocab_size()!=vocab:
        raise ValueError('requested train-only vocabulary size not attained')
    tokenizer.save(str(output/'tokenizer-a.json'))
    hashes=verify_checkpoint(Path(asset))
    qwen=AutoTokenizer.from_pretrained(asset,local_files_only=True,trust_remote_code=False)
    selection={'train':docs['train'], 'validation':choose_documents(docs['validation'],8,20261108),
               'test':choose_documents(docs['test'],16,20261108)}
    metadata=dict(schema='tide-language-data-v1',source=source_lock,sequence_length=length,
        selected_documents={s:[{k:d[k] for k in ('id','sha256','utf8_bytes','title')}
                               for d in rows] for s,rows in selection.items()},
        scope='natural-text subset; first two chunks per heldout document, independent official splits; no chat template',
        train_sampling='counter-seeded replacement over fixed hash-selected chunk pool',
        state_reset='fresh BOS per chunk, no cross-chunk or cross-document state',
        qwen_revision='7ae557604adf67be50417f59c2c2f167def9a775',
        tokenizer_a_sha256=digest(output/'tokenizer-a.json'),tokenizer_b_sha256=hashes['tokenizer.json'],routes={})
    for route,encode,bos,eos,pad,limit in (
        ('a',lambda text:tokenizer.encode(text).ids,1,2,0,4096),
        ('b',lambda text:qwen.encode(text,add_special_tokens=False),qwen.eos_token_id,qwen.eos_token_id,qwen.pad_token_id,512)):
        tensors,info={},{}
        for split,articles in selection.items():
            blocks=[]
            for doc in articles:
                tokens=encode(doc['text'])+[eos]
                for offset in range(0,len(tokens),length):
                    if split!='train' and offset>=2*length:break
                    x,y=shifted_block(tokens[offset:offset+length],length=length,bos=bos,pad=pad)
                    blocks.append(dict(x=x,y=y,document_id=doc['id'],offset=offset,
                                       id=f'{doc["id"]}:{offset}'))
            available=len(blocks)
            if split=='train':
                blocks=sorted(blocks,key=lambda b:text_hash('pool:'+b['id']))[:limit]
            x=torch.tensor([b['x'] for b in blocks],dtype=torch.long)
            y=torch.tensor([b['y'] for b in blocks],dtype=torch.long)
            tensors[split]=dict(input_ids=x,labels=y,document_ids=[b['document_id'] for b in blocks],
                                block_ids=[b['id'] for b in blocks])
            info[split]=dict(available_blocks=available,blocks=len(blocks),valid_tokens=int((y!=-100).sum()),
                document_ids=sorted({b['document_id'] for b in blocks}),
                block_ids_sha256=text_hash(json.dumps([b['id'] for b in blocks])))
        torch.save(tensors,output/f'{route}-tokens.pt')
        metadata['routes'][route]=dict(file=f'{route}-tokens.pt',sha256=digest(output/f'{route}-tokens.pt'),
            vocab_size=vocab if route=='a' else len(qwen),bos=bos,eos=eos,pad=pad,splits=info)
    atomic_json(output/'data-lock.json',metadata)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['extract','tokenize'])
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset-path',type=Path)
    a=p.parse_args()
    result=extract(a.input,a.output) if a.action=='extract' else tokenize(a.input,a.output,a.asset_path)
    print(json.dumps({k:result[k] for k in ('schema','article_counts','sequence_length') if k in result}))


if __name__=='__main__':main()
