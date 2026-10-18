"""Standalone protocol; historical records are provenance, never runtime dependencies."""
import argparse,json,shutil
from pathlib import Path
from runtime.utils import ROOT,dump,sha,now


def source_hashes():
    result={}
    for folder in ('repair','runtime','evaluation_c40','src','hs_connectors','scripts'):
        result.update({str(f):sha(f) for f in (ROOT/folder).rglob('*.py')})
    result.update({str(f):sha(f) for f in ROOT.glob('*.py')})
    return result


def build():
    p=json.loads((ROOT/'assets/protocol_template.json').read_text())
    p.update(standalone_bundle='residualV3_main_v1',utc=now(),source_files=source_hashes(),
        data_source=str(ROOT/'assets/data/regenerated.jsonl'),basis_path=str(ROOT/'assets/repair_basis.safetensors'),
        m_ids_path=str(ROOT/'assets/m_ids.json'),head_config_path=str(ROOT/'assets/head_config.json'),
        panel_paths={n:str(ROOT/'assets/panels'/f'{n}.jsonl') for n in ('train','warmup','final_C1','dev')})
    p['asset_sha256']={str(f):sha(f) for f in (ROOT/'assets').rglob('*') if f.is_file() and f.name!='preregistration.json'}
    p['dependency_sha256']={str(f):sha(f) for f in (ROOT/'runtime/body_only_draft').iterdir() if f.is_file()}
    external=json.loads((ROOT/'provenance/external_dependencies.json').read_text())
    p['dependency_sha256'].update(external['sha256'])
    p['standalone_lineage']=dict(original_protocol_sha256=sha(ROOT/'provenance/original_run/preregistration.json'),
        original_protocol_copy=str(ROOT/'provenance/original_run/preregistration.json'))
    return p


def read(path):
    p=json.loads(Path(path).read_text())
    assert p['standalone_bundle']=='residualV3_main_v1' and p['experiment']=='d1_dual_feedback_ab_v1'
    assert p['repair_initialization']=='random_from_scratch'
    assert p['train_arms']==['LMHEAD','EMBEDDING'] and p['architecture']['depth']==3 and p['architecture']['q_topk']==256
    assert p['soft_temperature']==1 and p['architecture']['dose_max']==.8
    assert not any(k in p for k in ('initial_checkpoint','resume_checkpoint','optimizer_checkpoint'))
    for mapping in ('source_files','asset_sha256','dependency_sha256'):
        for name,h in p[mapping].items():assert sha(name)==h,name
    assert sha(p['data_source'])==p['data_source_sha256']
    panels={}
    for name,f in p['panel_paths'].items():
        assert sha(f)==p['prompt_files_sha256'][name]
        panels[name]=[json.loads(s) for s in Path(f).read_text().splitlines() if s]
    assert len(panels['train'])==128 and len(panels['final_C1'])==20
    c40=ROOT/'assets/panels/final_C40.jsonl'
    assert sha(c40)=='c31d8053e89107a878b4c43f2476af8f5d85e820c8c2a1a979dc38a71a6d2cff'
    panels['final_C40']=[json.loads(s) for s in c40.read_text().splitlines() if s]
    assert len(panels['final_C40'])==300
    for name in ('dev','warmup','final_C1','final_C40'):
        for field in ('source_index','group_key','prompt_sha256'):
            assert not {r[field] for r in panels['train']}&{r[field] for r in panels[name]}
    assert p['test']['concurrency']==1 and sha(p['basis_path'])==p['basis_sha256']
    return p


def native_protocol():
    p=build();checkpoints={}
    for arm in ('LMHEAD','EMBEDDING'):
        f=ROOT/'assets/native'/f'{arm}.safetensors';original=ROOT/'assets/checkpoints'/f'{arm}_epoch_03.safetensors'
        checkpoints[arm]=dict(path=str(f),sha256=sha(f),original=str(original),original_sha256=sha(original),
            head_config_path=p['head_config_path'],head_config_sha256=sha(p['head_config_path']))
    p.update(native_arms=['RAW','LMHEAD','EMBEDDING'],native_configs={'LMHEAD':{},'EMBEDDING':{}},native_checkpoints=checkpoints)
    dump(ROOT/'assets/native/preregistration.json',p)
    return p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);p=build()
    dump(out/'head_config.json',p['architecture'])
    for f in source_hashes():
        source=Path(f);dest=out/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    dump(out/'preregistration.json',p);read(out/'preregistration.json')
    dump(out/'pipeline_status.json',dict(status='prepared',utc=now()));print(out)
if __name__=='__main__':main()
