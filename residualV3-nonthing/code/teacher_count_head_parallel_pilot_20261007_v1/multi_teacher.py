"""Two complete target replicas, deterministic output order, transient hidden."""
import hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def install(log_path, replicas):
    import openai,httpx2,torch
    import runtime.vllm_hidden as hidden
    from hs_connectors import FileTransfer
    Base=hidden.OnlineHiddenProvider
    assert len(replicas)==2
    log_path=Path(log_path)
    class MultiTeacherProvider(Base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert str(self.client.base_url).rstrip('/')==replicas[0]['endpoint'].rstrip('/')
            assert str(self.root)==replicas[0]['root']
            self.roots=[hidden.require_tmpfs(r['root']) for r in replicas]
            assert all(r.is_dir() for r in self.roots)
            self.clients=[self.client]
            for r in replicas[1:]:
                http=openai.DefaultHttpxClient(limits=httpx2.Limits(max_connections=24,max_keepalive_connections=0),trust_env=False)
                client=openai.OpenAI(base_url=r['endpoint'],api_key='EMPTY',max_retries=0,http_client=http)
                assert client.models.list().data[0].id==self.model
                self.clients.append(client)
            self.transfers=[FileTransfer(r) for r in self.roots]
            # Per-replica concurrency is unchanged at 16; 32 total pending requests.
            self.pool.shutdown(wait=True);self.prefetch=32;self.pool=ThreadPoolExecutor(max_workers=32)
            self.pending_count=[0,0];self.pending_tokens=[0,0];self.seconds_per_token=[.00015,.00015]
            self.replica_requests=[0,0];self.replica_deleted=[0,0];self.fallbacks=0
        def _fill(self):
            while len(self.queue)<self.prefetch:
                example=next(self.source,None)
                if example is None:break
                with self.counter_lock:
                    available=[i for i in range(2) if self.pending_count[i]<16]
                    assert available
                    size=len(example['input_ids'])
                    i=min(available,key=lambda k:((self.pending_tokens[k]+size)*self.seconds_per_token[k],self.pending_count[k],k))
                    self.pending_count[i]+=1;self.pending_tokens[i]+=size
                self.queue.append((example['input_ids'],self.pool.submit(self.fetch_on_endpoint,example,i)))
        def fetch_on_endpoint(self,example,i):
            with self.counter_lock:
                self.inflight+=1;self.peak_inflight=max(self.peak_inflight,self.inflight)
            try:
                try:return self.fetch_from(example,i)
                except (openai.APIConnectionError,openai.InternalServerError,OSError) as error:
                    # Read-only fallback during actual endpoint failure, never an optimizer retry.
                    with self.counter_lock:
                        self.fallbacks+=1
                        with log_path.open('a') as stream:stream.write(json.dumps(dict(event='endpoint_failure_fallback',failed_replica=i,error=repr(error),unix_time=time.time()))+'\n')
                    return self.fetch_from(example,1-i)
            finally:
                with self.counter_lock:
                    self.pending_count[i]-=1;self.pending_tokens[i]-=len(example['input_ids']);self.inflight-=1
        def fetch_from(self,example,i):
            from speculators.data_generation.vllm_client import generate_hidden_states
            started=time.perf_counter();handle=generate_hidden_states(self.clients[i],self.model,{'input_ids':example['input_ids']},timeout=180,max_retries=0)
            rpc_seconds=time.perf_counter()-started;path=Path(handle).resolve();assert path.parent==self.roots[i]
            with self.counter_lock:self.requests+=1;self.replica_requests[i]+=1
            transfer_seconds=check_seconds=copy_seconds=0.
            try:
                t=time.perf_counter();payload=self.transfers[i].get_generated(str(path));transfer_seconds=time.perf_counter()-t
                if payload is None:raise RuntimeError('missing generated hidden')
                t=time.perf_counter();ids,h=payload['token_ids'],payload['hidden_states']
                assert ids.tolist()==example['input_ids']
                assert h.shape==(len(example['input_ids']),self.layers,self.width)
                assert h.dtype==torch.bfloat16 and torch.isfinite(h).all()
                check_seconds=time.perf_counter()-t;t=time.perf_counter();result=h.clone();copy_seconds=time.perf_counter()-t
                return result
            finally:
                path.unlink(missing_ok=True);Path(str(path)+'.lock').unlink(missing_ok=True)
                elapsed=time.perf_counter()-started
                record=dict(event='hidden_fetch',replica=i,tokens=len(example['input_ids']),rpc_seconds=rpc_seconds,
                            transfer_seconds=transfer_seconds,check_seconds=check_seconds,copy_seconds=copy_seconds,total_seconds=elapsed,
                            token_ids_sha256=hashlib.sha256(json.dumps(example['input_ids']).encode()).hexdigest(),unix_time=time.time())
                with self.counter_lock:
                    self.deleted+=1;self.replica_deleted[i]+=1
                    self.seconds_per_token[i]=.9*self.seconds_per_token[i]+.1*elapsed/max(1,len(example['input_ids']))
                    with log_path.open('a') as stream:stream.write(json.dumps(record)+'\n')
        def close(self):
            super().close()
            for client in self.clients[1:]:client.close()
    hidden.OnlineHiddenProvider=MultiTeacherProvider
