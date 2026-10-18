"""One CPU writer; window-boundary reader. No per-block syscalls or GPU sync."""
import mmap,struct
NAMES=('captures','replays','fallbacks','eager')
class ExecutionCounters:
    def __init__(self,path):
        self.file=open(path,'r+b');self.buffer=mmap.mmap(self.file.fileno(),32)
    def inc(self,name):
        offset=8*NAMES.index(name)
        value=struct.unpack_from('<Q',self.buffer,offset)[0]
        struct.pack_into('<Q',self.buffer,offset,value+1)
def snapshot(path):
    # Called only outside measured windows after all requests have returned.
    with open(path,'rb') as f:values=struct.unpack('<4Q',f.read(32))
    return dict(zip(NAMES,values))
def delta(before,after):return {key:after[key]-before[key] for key in NAMES}
