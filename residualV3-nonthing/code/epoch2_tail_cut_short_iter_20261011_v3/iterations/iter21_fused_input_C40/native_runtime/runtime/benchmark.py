"""Concurrent HTTP requests; stock full-vocabulary sampling parameters."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .http import _http_json
from .utils import healthy

def request(model,prompt,index,temp,length):
    return dict(model=model,messages=prompt['messages'],max_tokens=length,stream=False,temperature=temp,
        seed=42+index,top_p=1.,top_k=-1,frequency_penalty=0.,presence_penalty=0.,repetition_penalty=1.)

def parallel_requests(endpoint,model,prompts,temp,length,concurrency):
    def one(index,prompt):
        start=time.perf_counter_ns();wall=time.time_ns()
        response=_http_json(endpoint+'/v1/chat/completions',request(model,prompt,index,temp,length),timeout=1800)
        end=time.perf_counter_ns()
        return dict(index=index,temperature=temp,seed=42+index,start_unix_ns=wall,
            end_unix_ns=time.time_ns(),elapsed_seconds=(end-start)/1e9,response=response)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(one,i,prompt) for i,prompt in enumerate(prompts)]
        rows=[future.result() for future in as_completed(futures)]
    return sorted(rows,key=lambda row:row['index'])
