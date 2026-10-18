import argparse,json
from pathlib import Path
from runtime.utils import dump,sha,now

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
    src=out/'C40/summary.json';s=json.loads(src.read_text());rows=s['windows']
    assert s['status']=='complete' and len(rows)==12
    assert all(w['speed_evidence'] and w['isolation_observed'] and w['peak_http_inflight']==40 for w in rows)
    pairs=[]
    lines=['# GPU4 C40：RAW / LMHEAD / EMBEDDING','',
      '沿用GPU5 C1的epoch3权重；既有300条不同prompt，40并发，seed42..341。T0/T1，两轮正反顺序。',
      '每请求最多4096输出、总长度8192、显存比例0.93，CUDA Graph。40为客户端并发上限，结束阶段并发自然下降；不等于全程40个GPU活跃序列。',
      '测试集有历史开发使用记录；两轮不是独立样本。本表只在GPU4内部比较，不把跨卡C1/C40差异归因于并发。','',
      '|T|轮|模型|接受长度|token/s|相对RAW接受长度|相对RAW吞吐|',
      '|---|---|---|---:|---:|---:|---:|']
    for t in (0,1):
      for rnd in (1,2):
        cells={w['arm']:w for w in rows if w['T']==t and w['round']==rnd}
        assert set(cells)=={'RAW','LMHEAD','EMBEDDING'}
        raw=cells['RAW']
        for arm in ('RAW','LMHEAD','EMBEDDING'):
            w=cells[arm];ar=w['native_metrics']['acceptance_length']/raw['native_metrics']['acceptance_length'];tr=w['tokens_per_second']/raw['tokens_per_second']
            pairs.append(dict(T=t,round=rnd,arm=arm,acceptance_ratio=ar,throughput_ratio=tr))
            lines.append(f"|{t}|{rnd}|{arm}|{w['native_metrics']['acceptance_length']:.6f}|{w['tokens_per_second']:.2f}|{(ar-1)*100:+.2f}%|{(tr-1)*100:+.2f}%|")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(status='complete',utc=now(),summary_sha256=sha(src),paired=pairs))
if __name__=='__main__':main()
