"""Reuse validated epoch2 exports; preserve all original long-training products."""
import argparse,hashlib,json,sys,re,unicodedata
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer

CODE=Path(__file__).resolve().parent
PROJECT=CODE.parents[2]
sys.path.insert(0,str(CODE/'native_runtime'))
from runtime.utils import dump,sha,now
from device import assignment

def normalize(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).strip()).casefold()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    priorroot=PROJECT/'residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1'
    prior=json.loads((priorroot/'native_protocol.json').read_text())
    arms=['DFLASH','FULL15','Q45_CUT','Q45_LATENT_CUT']
    source='E2_JOINT_HEAD';body='E2_DFLASH'
    aux=PROJECT/'residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/final/post_repair_aux_comparison.pt'
    state=torch.load(aux,map_location='cpu',weights_only=True)
    assert state['steps']==6958 and state['format']=='two_detached_online_MLPs_v1'
    p={k:v for k,v in prior.items() if k not in ('source_files',)}
    mapping={a:body if a=='DFLASH' else source for a in arms}
    for k in ('native_checkpoints','body_models','body_checksums','architecture_by_arm','training_lineage_by_arm'):
        p[k]={a:prior[k][v] for a,v in mapping.items()}
    p.update(native_arms=arms,epoch_by_arm={a:2 for a in arms},native_configs={a:{} for a in arms},
        runtime=str(CODE/'native_runtime'),repair_enabled_by_arm={a:a!='DFLASH' for a in arms},
        optimization_by_arm={a:'small_tile' if a!='DFLASH' else 'baseline' for a in arms},
        head_structure_mode_by_arm={a:'lean_combined' if a!='DFLASH' else 'baseline' for a in arms},
        tail_mode_by_arm={a:'PIPELINE' if a.endswith('_CUT') else 'FULL15' for a in arms},
        auxiliary_branch_by_arm={a:'Q45_REPAIR_LATENT_MLP' if a=='Q45_LATENT_CUT' else 'Q45_MLP' for a in arms},
        repair_prefix_length_by_arm={'DFLASH':0,'FULL15':15,'Q45_CUT':None,'Q45_LATENT_CUT':None},
        gpu_by_concurrency={'C1':6,'C40':6},auxiliary_checkpoint=str(aux),auxiliary_checkpoint_sha256=sha(aux),
        predictor_path=str(root/'calibration/predictor.json'),predictor_sha256='pending_native_calibration',
        cutoff_enabled=True,auxiliary_predictors_enabled=True,auxiliary_frozen=True,
        target_graph_mode='FULL_AND_PIECEWISE',kv_cache_gib_by_phase={'smoke':{'C1':52,'C40':38},'formal':{'C1':52,'C40':44}},
        scope='Epoch2 trained post-repair readouts; one-position delayed causal repair/verification cutoff; actual native AL/TPS; fixed formal GSM8K test panels')
    sources={str(f):sha(f) for f in CODE.rglob('*.py')}
    p['source_files']=sources
    dump(root/'source_sha.json',sources);dump(root/'native_protocol.json',p)
    for c in (1,40):
        lane=root/f'C{c}';lane.mkdir(exist_ok=True)
        if not (lane/'native_protocol.json').exists():(lane/'native_protocol.json').symlink_to(root/'native_protocol.json')
    # A separate train-split panel; formal test text cannot tune any threshold.
    cal=root/'calibration';rows=pq.read_table(cal/'gsm8k_train.parquet').to_pylist()
    test={normalize(v['messages'][0]['content'].replace('\\n','\n').split('\nPlease reason')[0]) for v in
          [json.loads(s) for s in Path(p['panel_paths']['final_C40']).read_text().splitlines()]}
    test_ids={v['id'] for v in [json.loads(s) for s in Path(p['panel_paths']['final_C40']).read_text().splitlines()]}
    tokenizer=AutoTokenizer.from_pretrained(p['target_model'],local_files_only=True,trust_remote_code=True)
    selected=[];seen=set()
    for i,row in enumerate(rows):
        key=normalize(row['question'])
        if key in test or key in seen:continue
        seen.add(key)
        prompt=row['question']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
        ids=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert isinstance(ids,list) and all(isinstance(v,int) for v in ids)
        if len(ids)>3904:continue
        selected.append(dict(id=f'gsm8k-train:{i}',probe_index=i,probe_split='calibration' if len(selected)<64 else 'holdout',prompt_token_ids=ids))
        if len(selected)==96:break
    assert len(selected)==96 and not {r['id'] for r in selected}&test_ids
    panel=cal/'panels.jsonl';panel.write_text(''.join(json.dumps(r)+'\n' for r in selected))
    policy=dict(calibration_policy=dict(minimum_repair=8,patience=2,thresholds=[0.,0.]),policies={})
    dump(cal/'initial_policy.json',policy)
    cp=dict(p,physical_gpu=6,gpu_uuid=assignment(6)['uuid'],checkpoint=p['native_checkpoints']['FULL15'],
        body_model=p['body_models']['FULL15'],body_checksums=p['body_checksums']['FULL15'],
        panel_path=str(panel),probe_panel_sha256=sha(panel),minimum_free_mib=78000,kv_cache_gib=52,
        requested_gpu_memory_utilization=.92,maximum_blocks_per_prompt=8,capture_tokens=192,
        calibration_prompts=64,holdout_prompts=32,formal_panel_question_overlap=0)
    dump(cal/'preregistration.json',cp)
    micro=dict(p,gpu_uuid=assignment(6)['uuid'],checkpoint=p['native_checkpoints']['FULL15'])
    dump(root/'preregistration.json',micro)
    (root/'RUN_GUIDE_CN.txt').write_text('第二轮已训练预测头尾部截断；冻结DFlash/repair/预测头。\n目前仅卡6空闲，先卡6顺序C1=200、C40=1319；T0/T1；block16含15个proposal；max2096；nonthinking；BF16；同卡target+draft。\n四组：单训DFlash、完整修复、q45预测截断、q45+repair末层预测截断。截断同时缩短修复与target验证前缀。\nGSM8K train 64题校准+32题只作检查，与正式test问句不重叠。至少修8位置、连续2次低前缀生存概率、一位置延迟；仅在校准集选阈值，固定全修路径保留率>=99.5%不是正式成绩保证。\n权重复用现有验证导出，不新建大权重；即时日志single_card_C1_formal.log、single_card_C40_formal.log；校准calibration.log；门槛micro.log。\n')
    dump(root/'status.json',dict(status='prepared',utc=now()))
    print('PREPARED',root,flush=True)

if __name__=='__main__':main()
