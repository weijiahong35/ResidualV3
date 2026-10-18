"""Owned-card allocation handoff; never claims a foreign GPU or kills processes."""
import argparse,json,os,time
from pathlib import Path

def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value));tmp.replace(path)

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False

def release_owned_keeper(p,arm):
    value=p.get('handoff_keepers',{}).get(arm)
    if not value:return
    d=Path(value['root']);write(d/'release.json',dict(reason='replacement learner CUDA context initialized',pid=os.getpid()))
    deadline=time.monotonic()+60
    while not(d/'released.json').exists():
        assert alive(value['pid']),'Owned keeper exited before handoff'
        assert time.monotonic()<deadline,'Owned keeper release timed out'
        time.sleep(.1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--uuid',required=True);ap.add_argument('--old-worker',type=int,required=True)
    a=ap.parse_args();r=a.root
    os.environ['CUDA_VISIBLE_DEVICES']=a.uuid
    import torch
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.98)
    chunks=[torch.empty(512*2**20,device='cuda',dtype=torch.uint8)]
    write(r/'ready.json',dict(pid=os.getpid(),uuid=a.uuid,old_worker=a.old_worker,
                            initial_bytes=512*2**20))
    while not(r/'stop.json').exists():
        if (r/'release.json').exists():
            chunks.clear();torch.cuda.empty_cache()
            # A live replacement CUDA context now owns this card.
            write(r/'released.json',dict(pid=os.getpid(),replacement=json.loads((r/'release.json').read_text()),
                                        allocated_bytes=torch.cuda.memory_allocated()))
            while not(r/'stop.json').exists():time.sleep(.2)
            break
        if not alive(a.old_worker):
            free,_=torch.cuda.mem_get_info()
            if free>3*2**30:
                # Retain a real buffer until the replacement initializes CUDA.
                amount=min(2**30,free-2*2**30)
                try:chunks.append(torch.empty(amount,device='cuda',dtype=torch.uint8))
                except torch.OutOfMemoryError:torch.cuda.empty_cache()
        write(r/'status.json',dict(pid=os.getpid(),old_worker_alive=alive(a.old_worker),
            allocated_gib=torch.cuda.memory_allocated()/2**30,free_gib=torch.cuda.mem_get_info()[0]/2**30))
        time.sleep(.1)
    write(r/'complete.json',dict(status='released',foreign_processes_untouched=True))

if __name__=='__main__':main()
