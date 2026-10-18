"""Immutable token/mask-only data; ordered parallel reading, no hidden cache."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'runtime_bundle'))
from runtime.utils import dump, sha, now

RECORD = np.dtype([('offset','<u8'),('length','<u4'),('source_line','<u4'),
                   ('group','V32'),('row_sha','V32')])


def eligible_mask(mask, block=16):
    """Anchor and all proposals supervised; preserve the official final-block exclusion."""
    mask = np.asarray(mask, dtype=np.uint8)
    result = np.zeros(len(mask), dtype=np.bool_)
    if len(mask) > block:
        cumulative = np.concatenate(([0], np.cumsum(mask, dtype=np.int32)))
        result[:-block] = (cumulative[block:]-cumulative[:-block])[:-1] == block
    return result


def prepare_shard(item):
    index, name, destination = item
    source = Path(name); out = Path(destination)/f'shard_{index:03d}'
    out.mkdir(exist_ok=False)
    records=[]; groups=set(); rejected=Counter(); offset=0; source_hash=hashlib.sha256()
    with source.open('rb') as stream, (out/'tokens.u32').open('wb') as tokens, (out/'masks.u8').open('wb') as masks:
        for line_index, line in enumerate(stream):
            source_hash.update(line)
            if not line.strip(): continue
            row=json.loads(line); meta=row['metadata']
            reason=None
            if not meta.get('conversation_complete'): reason='not_final_conversation_row'
            elif meta.get('finish_reason') not in ('stop','length'): reason='unsuccessful_finish'
            elif meta['sampling_params'].get('chat_template_kwargs',{}).get('enable_thinking') is not False: reason='thinking_mismatch'
            if reason: rejected[reason]+=1; continue
            values=np.asarray(row['input_ids']); mask=np.asarray(row['loss_mask'])
            group=bytes.fromhex(meta['source']['pool']['group_sha256'])
            if group in groups: reason='duplicate_within_shard'
            elif len(group)!=32: reason='group_format'
            elif values.ndim!=1 or not 17<=len(values)<=4096 or mask.shape!=values.shape: reason='length_or_shape'
            elif values.dtype.kind not in 'iu' or np.any(values<0) or np.any(values>=151936): reason='token_range'
            elif np.any((mask!=0)&(mask!=1)): reason='mask_values'
            elif not eligible_mask(mask).any(): reason='no_complete_supervised_block'
            if reason: rejected[reason]+=1; continue
            groups.add(group)
            tokens.write(values.astype('<u4',copy=False).tobytes()); masks.write(mask.astype('u1',copy=False).tobytes())
            records.append((offset,len(values),line_index,group,hashlib.sha256(line).digest())); offset+=len(values)
    np.save(out/'records.npy',np.array(records,dtype=RECORD),allow_pickle=False)
    result=dict(index=index,source=str(source),source_sha256=source_hash.hexdigest(),directory=str(out),
                records=len(records),tokens=offset,rejected=dict(rejected),
                files={str(out/name):sha(out/name) for name in ['tokens.u32','masks.u8','records.npy']})
    dump(out/'manifest.json',result)
    return result


def prepare(plan):
    destination=Path(plan['data_root'])
    completion=Path(plan['generation_root'])/'completion.json'
    verified=json.loads(completion.read_text())
    assert verified['all_processed'] and verified['remaining_conversations']==0
    ready=destination/'manifest.json'
    if ready.exists():
        result=json.loads(ready.read_text()); assert result['status']=='complete'
        assert result['completion_sha256']==sha(completion)
        return result
    paths=json.loads((completion.parent/'training_data_paths.json').read_text())['train']
    assert len(paths)==85 and len(set(paths))==85
    dump(destination/'status.json',dict(status='preparing_token_only_data',total_shards=len(paths),utc=now()))
    items=[(index,name,str(destination)) for index,name in enumerate(paths)]
    # Each process writes its own shard, never sends token arrays through a pipe.
    with ProcessPoolExecutor(max_workers=plan['data_loader_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        shards=[]
        for row in pool.map(prepare_shard,items):
            shards.append(row)
            dump(destination/'status.json',dict(status='preparing_token_only_data',shards=len(shards),total_shards=len(paths),utc=now()))
            print(now(),'DATA_SHARD',len(shards),'/',len(paths),'rows',row['records'],flush=True)
    groups=set(); catalog=[]; duplicates=0
    for shard in shards:
        records=np.load(Path(shard['directory'])/'records.npy',mmap_mode='r',allow_pickle=False)
        for index,row in enumerate(records):
            key=bytes(row['group'])
            if key in groups: duplicates+=1; continue
            groups.add(key); catalog.append((shard['index'],index,int(row['length'])))
    catalog=np.array(catalog,dtype='<u4')
    assert len(catalog)==len(groups)>0
    np.save(destination/'catalog.npy',catalog,allow_pickle=False)
    result=dict(status='complete',count=len(catalog),unique_groups=len(groups),tokens=int(catalog[:,2].astype(np.uint64).sum()),
                catalog=str(destination/'catalog.npy'),catalog_sha256=sha(destination/'catalog.npy'),
                source_shards=shards,completion_path=str(completion),completion_sha256=sha(completion),
                source_accounting=verified,duplicates_across_shards=duplicates,max_length=int(catalog[:,2].max()),
                includes_valid_length_truncated_prefixes=True,supervised_block_size=16,
                no_hidden_or_logits_persisted=True,validation_created=False,all_valid_training_conversation_groups=True,utc=now())
    dump(ready,result); dump(destination/'status.json',dict(status='complete',count=len(catalog),tokens=result['tokens'],utc=now()))
    return result


class IndexedData:
    def __init__(self, manifest):
        self.manifest=manifest; self.catalog=np.load(manifest['catalog'],mmap_mode='r',allow_pickle=False)
        self.parts=[]
        for shard in manifest['source_shards']:
            path=Path(shard['directory'])
            self.parts.append((np.load(path/'records.npy',mmap_mode='r',allow_pickle=False),
                               np.memmap(path/'tokens.u32',mode='r',dtype='<u4'),
                               np.memmap(path/'masks.u8',mode='r',dtype='u1')))

    def __len__(self): return len(self.catalog)

    def __getitem__(self,index):
        part,row,length=map(int,self.catalog[index]); records,tokens,masks=self.parts[part]
        record=records[row]; start=int(record['offset']); end=start+length
        return dict(index=int(index),input_ids=tokens[start:end].tolist(),loss_mask=masks[start:end].tolist(),group_sha256=bytes(record['group']).hex())

    def read_order(self,indices,workers=8,prefetch=128):
        # Bounded submit queue rather than Executor.map's eager submission of 800k rows.
        from collections import deque
        iterator=iter(indices); queue=deque()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def fill():
                while len(queue)<prefetch:
                    index=next(iterator,None)
                    if index is None:break
                    queue.append(pool.submit(self.__getitem__,int(index)))
            fill()
            while queue:
                value=queue.popleft().result(); fill(); yield value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    prepare(json.loads(parser.parse_args().plan.read_text()))
