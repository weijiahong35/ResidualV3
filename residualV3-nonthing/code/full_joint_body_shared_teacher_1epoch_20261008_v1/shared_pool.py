"""Fixed shared CPU storage; ownership ends only after both learners finish copying."""
from types import SimpleNamespace
import torch
import numpy as np
from data_source import eligible_mask

class SharedPool:
    def __init__(self,slots=2,budget=65536,width=4096,layers=6):
        self.slots=[];self.free=list(range(slots));self.budget=budget
        for _ in range(slots):
            values=dict(ids=torch.empty((1,budget),dtype=torch.long),docs=torch.empty((1,budget),dtype=torch.long),positions=torch.empty((1,budget),dtype=torch.long),eligible=torch.empty(budget,dtype=torch.long),aux=torch.empty((1,budget,(layers-1)*width),dtype=torch.bfloat16),final=torch.empty((budget,width),dtype=torch.bfloat16))
            for value in values.values():value.share_memory_()
            self.slots.append(values)
    def acquire(self,rows):
        assert self.free,'No shared slot available'
        slot=self.free.pop(0);values=self.slots[slot]
        total=sum(len(row['input_ids']) for row in rows);assert 0<total<=self.budget
        values['ids'].zero_();values['docs'].fill_(-1);values['positions'].zero_()
        if total<self.budget:values['aux'][:,total:].zero_()
        choices=[];offset=0
        for document,row in enumerate(rows):
            n=len(row['input_ids']);values['ids'][0,offset:offset+n].copy_(torch.tensor(row['input_ids']))
            values['docs'][0,offset:offset+n].fill_(document);values['positions'][0,offset:offset+n].copy_(torch.arange(n))
            choices.extend((np.flatnonzero(eligible_mask(row['loss_mask']))+offset).tolist());offset+=n
        assert choices
        values['eligible'][:len(choices)].copy_(torch.tensor(choices))
        selected=dict(values,eligible=values['eligible'][:len(choices)],final=values['final'][:total])
        return slot,SimpleNamespace(**selected,offset=total)
    def release(self,slot):
        assert slot not in self.free;self.free.append(slot)

def gate(reference):
    pool=SharedPool(slots=2,budget=128,width=4,layers=6)
    rows=[dict(input_ids=list(range(31)),loss_mask=[0]*5+[1]*26),dict(input_ids=list(range(45)),loss_mask=[0]*7+[1]*38)]
    for _ in range(4):
        slot,host=pool.acquire(rows);old=reference.allocate_host(rows,128,width=4,layers=6,pinned=False)
        def fetch(row):return torch.arange(len(row['input_ids'])*24,dtype=torch.float32).reshape(-1,6,4).to(torch.bfloat16)
        reference.fill_host(host,rows,fetch);reference.fill_host(old,rows,fetch)
        for name in ['ids','docs','positions','eligible','final']:
            assert torch.equal(getattr(host,name),getattr(old,name)),name
        assert torch.equal(host.aux[:,:host.offset],old.aux[:,:old.offset])
        assert bool((host.aux[:,host.offset:]==0).all())
        assert all(getattr(host,k).is_shared() for k in ['ids','docs','positions','eligible','aux','final'])
        pool.release(slot)
        rows=rows[::-1]
    return dict(passed=True,exact_input_metadata_and_hidden_match=True,reuse_cycles=4,live_padding_zeroed=True,two_slots_reused=True)
