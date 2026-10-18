"""Single official DP endpoint; 32 total ordered online requests."""
import time,json,hashlib,os,uuid
from threading import BoundedSemaphore
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ordered_scheduler import OrderedRequests

def install(log_path,replica):
    import openai,httpx2,torch
    import runtime.vllm_hidden as hidden
    if not hasattr(hidden,'_throughput_original_provider'):
        hidden._throughput_original_provider=hidden.OnlineHiddenProvider
    Base=hidden._throughput_original_provider;log_path=Path(log_path)
    class DPProvider(Base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert str(self.client.base_url).rstrip('/')==replica['endpoint'].rstrip('/') and str(self.root)==replica['root']
            endpoint=str(self.client.base_url);self.client.close();concurrency=int(os.environ.get('THROUGHPUT_REQUESTS','32'));assert concurrency in [32,64,128]
            self.reader_limit=int(os.environ.get('THROUGHPUT_READERS',str(concurrency)))
            self.readers=BoundedSemaphore(self.reader_limit)
            keepalive=concurrency if os.environ.get('THROUGHPUT_KEEPALIVE','0')=='1' else 0
            http=openai.DefaultHttpxClient(limits=httpx2.Limits(max_connections=concurrency+16,max_keepalive_connections=keepalive),trust_env=False)
            self.client=openai.OpenAI(base_url=endpoint,api_key='EMPTY',max_retries=0,http_client=http)
            self.pool.shutdown(wait=True);self.prefetch=concurrency;self.pool=ThreadPoolExecutor(max_workers=concurrency);self.roots=[self.root]
            self.ordered=OrderedRequests(self.pool,self.fetch,concurrency=concurrency,window=2*concurrency)
        def begin_epoch(self,examples):
            self.ordered.begin(examples)
        def __call__(self,example):
            return self.ordered.get(example)
        def close(self):
            self.ordered.close()
            self.client.close()
        def _fetch(self,example):
            from speculators.data_generation.vllm_client import generate_hidden_states
            start=time.perf_counter()
            from speculators.data_generation.vllm_client import extract_output
            response=self.client.completions.create(model=self.model,prompt=example['input_ids'],max_tokens=1,
                extra_body={'return_token_ids':True,'cache_salt':str(uuid.uuid4())},timeout=180)
            handle=extract_output(response,example['input_ids'])
            rpc=time.perf_counter()-start;path=Path(handle).resolve();assert path.parent==self.root
            with self.counter_lock:self.requests+=1
            read=check=copy=lock_wait=reader_wait=0.
            try:
                # Preserve the connector's writer-completion barrier and exact
                # safetensors decoder. Bound CPU decode/check/copy separately
                # from outstanding GPU/HTTP requests.
                from hs_connectors.transfer import wait_for_lock
                from safetensors.torch import load_file
                t=time.perf_counter();lock_path=Path(str(path)+'.lock')
                if lock_path.exists():wait_for_lock(str(lock_path),poll_interval=float(os.environ.get('THROUGHPUT_LOCK_POLL','.1')))
                lock_wait=time.perf_counter()-t
                t=time.perf_counter();self.readers.acquire();reader_wait=time.perf_counter()-t
                try:
                    t=time.perf_counter();assert path.exists();payload=load_file(str(path));read=time.perf_counter()-t
                    assert payload is not None;t=time.perf_counter();ids,h=payload['token_ids'],payload['hidden_states']
                    assert ids.tolist()==example['input_ids'] and h.shape==(len(example['input_ids']),self.layers,self.width)
                    assert h.dtype==torch.bfloat16 and torch.isfinite(h).all();check=time.perf_counter()-t
                    t=time.perf_counter()
                    # safetensors' returned Torch storage owns the mmap lifetime.
                    # Linux unlink removes the name; the live tensor remains valid.
                    # Verified by the CPU lifetime gate before enabling this path.
                    result=h if os.environ.get('THROUGHPUT_MMAP_OWNERSHIP','0')=='1' else h.clone()
                    copy=time.perf_counter()-t;return result
                finally:self.readers.release()
            finally:
                path.unlink(missing_ok=True);Path(str(path)+'.lock').unlink(missing_ok=True)
                record=dict(event='hidden_fetch',unique_request_cache_salt=True,teacher_dp_size=replica.get('dp_size',1),tokens=len(example['input_ids']),rpc_seconds=rpc,transfer_seconds=read,
                            lock_wait_seconds=lock_wait,reader_wait_seconds=reader_wait,reader_limit=self.reader_limit,check_seconds=check,copy_seconds=copy,total_seconds=time.perf_counter()-start,
                            mmap_ownership=os.environ.get('THROUGHPUT_MMAP_OWNERSHIP','0')=='1',lock_poll_seconds=float(os.environ.get('THROUGHPUT_LOCK_POLL','.1')),
                            token_ids_sha256=hashlib.sha256(json.dumps(example['input_ids']).encode()).hexdigest(),unix_time=time.time())
                with self.counter_lock:
                    self.deleted+=1
                    with log_path.open('a') as stream:stream.write(json.dumps(record)+'\n')
    hidden.OnlineHiddenProvider=DPProvider
