"""Real vLLM+online DFlash states: changing a future answer cannot alter drafts."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from .protocol import read
from .run import Session
from runtime.live_trainer import online_states
from runtime.utils import dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    p=read(a.protocol);torch.set_num_threads(4);torch.cuda.set_device(0)
    s=Session(p,a.output.parent/'causality_cleanup')
    try:
        row=s.train[0];first=row['loss_mask'].index(1);anchor_pos=first+4
        end=min(len(row['input_ids']),anchor_pos+64)
        one={k:list(row[k][:end]) for k in ('input_ids','loss_mask')}
        two={k:list(v) for k,v in one.items()}
        two['input_ids'][anchor_pos+1]=(two['input_ids'][anchor_pos+1]+17)%151936
        s.provider.begin_epoch([one,two])
        def fixed(mask,num_anchors,block_size):
            assert num_anchors==16
            return torch.full((16,),anchor_pos,device=mask.device,dtype=torch.long),torch.ones(16,device=mask.device,dtype=torch.bool)
        with patch('speculators.models.dflash.utils.select_anchors',fixed):
            h1,t1,y1,a1=online_states(s.target,s.draft,one,16,torch.device('cuda:0'))
            h2,t2,y2,a2=online_states(s.target,s.draft,two,16,torch.device('cuda:0'))
        assert torch.equal(a1,a2) and not torch.equal(y1[:,0],y2[:,0])
        assert torch.equal(h1,h2),'draft hidden reads answer beyond its verified anchor'
        assert torch.equal(t1[:,0],t2[:,0]),'target state at anchor reads future answer'
        assert not torch.equal(t1[:,1],t2[:,1]),'intervention failed to change later target state'
        dump(a.output,dict(status='passed',future_token_offset=1,anchor_position=anchor_pos,
            all_seven_draft_hidden_exact=True,target_anchor_hidden_exact=True,later_target_changed=True,
            online_target=True,online_dflash=True,artificial_sequence_validation_only=True))
    finally:s.close()


if __name__=='__main__':main()
