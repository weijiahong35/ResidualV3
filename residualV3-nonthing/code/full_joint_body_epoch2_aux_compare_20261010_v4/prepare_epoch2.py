"""Freeze epoch-two data/lineage and verify the source state without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

CODE=Path(__file__).resolve().parent
sys.path[:0]=[str(CODE),str(CODE.parent/'runtime_bundle')]
from full_checkpoint import atomic,prefix_sha
from runtime.utils import now


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();old=a.previous.resolve()
    p=json.loads((old/'protocol.json').read_text())
    assert json.loads((old/'status.json').read_text())['status']=='complete'
    pointer=json.loads((old/'latest_checkpoint.json').read_text())
    assert pointer['status']=='committed' and pointer['cursor']==p['pack_count']
    assert pointer['joint_and_body_same_data_cursor']
    old_proof=json.loads((old/'identity_proof.json').read_text())
    assert sha(old/'protocol.json')==old_proof['protocol_sha256']
    assert sha(p['data_order'])==old_proof['order_sha256']
    assert sha(p['pack_offsets'])==old_proof['plan_sha256']
    manifest=json.loads(Path(p['data_manifest']).read_text())
    assert manifest['status']=='complete' and manifest['count']==p['training_rows']
    assert sha(manifest['catalog'])==manifest['catalog_sha256']
    catalog=np.load(manifest['catalog'],mmap_mode='r',allow_pickle=False)
    order=np.random.default_rng(p['shuffle_seed']+1).permutation(len(catalog)).astype('<u4')
    assert len(np.unique(order))==len(catalog) and np.array_equal(np.sort(order),np.arange(len(catalog)))
    assert not np.array_equal(order,np.load(p['data_order'],mmap_mode='r',allow_pickle=False))
    lengths=catalog[order,2].astype(np.uint64)
    bounds=[0];tokens=0
    for index,length in enumerate(lengths):
        length=int(length)
        if tokens+length>65536:bounds.append(index);tokens=0
        tokens+=length
    bounds.append(len(order));bounds=np.asarray(bounds,dtype='<u8')
    starts=np.concatenate(([0],np.cumsum(lengths,dtype=np.uint64)))
    pack_sizes=starts[bounds[1:]]-starts[bounds[:-1]]
    assert 0<int(pack_sizes.min())<=int(pack_sizes.max())<=65536
    assert int(pack_sizes.sum())==p['input_tokens']
    np.save(root/'epoch_order.npy',order,allow_pickle=False)
    np.save(root/'pack_offsets.npy',bounds,allow_pickle=False)
    # Pin a read-only copy of the first epoch's pointer; source files are never written.
    atomic(root/'initial_checkpoint.json',pointer)
    p.update(job_root=str(root),variant='full_nonthinking_epoch2_shared_teacher',training_epoch=2,
        epochs=1,global_step_offset=pointer['cursor'],pack_count=len(bounds)-1,
        data_order=str(root/'epoch_order.npy'),pack_offsets=str(root/'pack_offsets.npy'),
        shuffle_seed=p['shuffle_seed']+1,initial_checkpoint_pointer=str(root/'initial_checkpoint.json'),
        initial_checkpoint_proof=old_proof,initialization='Restore epoch1 BF16 via FP32 masters, full Muon/Adam and RNG; no optimizer reset',
        lr_schedule='new epoch cycle: 1% linear warmup then cosine to zero, inherited optimizer moments',
        new_scope='One additional full epoch (epoch2), no validation or native evaluation',
        source_epoch_root=str(old),excluded_gpus_by_user=[1,2],teacher_dp_size=1,
        teacher_tp_size=1,training_data_parallel_size_per_arm=1,global_anchors_per_update_per_arm=1536,
        gradient_accumulation_packs=1,head_chunk_requested=1536,utc=now())
    atomic(root/'protocol.json',p)
    atomic(root/'teacher_input_protocol.json',dict(target_model=p['target_model']))
    atomic(root/'identity_proof.json',dict(protocol_sha256=sha(root/'protocol.json'),
        order_sha256=sha(root/'epoch_order.npy'),plan_sha256=sha(root/'pack_offsets.npy')))
    assets={name:sha(p[name]) for name in ['basis_path','head_config_path','data_manifest']}
    atomic(root/'lineage.json',dict(previous_epoch_root=str(old),previous_global_step=pointer['cursor'],
        source_proof=old_proof,assets_sha256=assets,initial_artifacts=pointer['artifacts'],
        protocol_change='epoch/order/schedule horizon/output paths; model, loss, pack and anchor budgets unchanged'))
    atomic(root/'data_preflight.json',dict(status='passed',training_rows=len(order),input_tokens=int(pack_sizes.sum()),
        pack_count=len(bounds)-1,shuffle_seed=p['shuffle_seed'],all_rows_exactly_once=True,
        min_pack_tokens=int(pack_sizes.min()),max_pack_tokens=int(pack_sizes.max()),
        no_cross_document_windows=True,token_only_data_reused=True,no_persistent_hidden_logits=True,utc=now()))
    print(now(),'EPOCH2_PLAN_READY',len(order),'rows',len(bounds)-1,'packs',flush=True)
    artifacts={}
    for arm,descriptor in pointer['artifacts'].items():
        assert descriptor['cursor']==pointer['cursor'] and descriptor['arm']==arm
        assert descriptor['full_masters_optimizers_rng'] and descriptor['actual_bf16_weights_reconstruct_exactly']
        assert all(descriptor[k]==v for k,v in old_proof.items())
        assert prefix_sha(descriptor['path'],descriptor['logical_bytes'])==descriptor['sha256']
        final=json.loads((old/'final'/f'{arm}_manifest.json').read_text())
        assert Path(final['path']).stat().st_size==final['bytes']
        assert sha(final['path'])==final['sha256']
        artifacts[arm]=dict(status='passed',source_checkpoint_sha256=descriptor['sha256'],
            source_final_bf16_sha256=final['sha256'],global_step=pointer['cursor'])
        print(now(),'SOURCE_CHECKPOINT_VERIFIED',arm,flush=True)
    previous_sources=json.loads((old/'source_sha.json').read_text())
    assert all(sha(path)==digest for path,digest in previous_sources.items())
    # Preserve dependency pins and include the copied continuation implementation.
    sources={path:digest for path,digest in previous_sources.items() if Path(path).parent!=old.parent.parent/'code/full_joint_body_shared_teacher_1epoch_20261008_v1'}
    for path in CODE.glob('*.py'):sources[str(path)]=sha(path)
    atomic(root/'source_sha.json',sources)
    atomic(root/'gates/launch_preflight.json',dict(status='passed',artifacts=artifacts,
        initial_weights_and_optimizer_files_checksums_verified=True,old_sources_unchanged=True,
        source_files=len(sources),full_actual_cuda_restore_hash_gate_at_startup=True,
        second_epoch_first_update_full_cuda_save_restore_required=True,utc=now()))
    atomic(root/'status.json',dict(status='prepared_waiting_service',training_epoch=2,
        initial_global_step=pointer['cursor'],training_rows=len(order),total_packs=len(bounds)-1,
        gpu_process_launched=False,utc=now()))


if __name__=='__main__':main()
