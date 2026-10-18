"""Inference-only predecessor lookup and branch concatenation, without arithmetic.

Keep the original LayerNorm and GEMM. This kernel only copies the same BF16
feature values into the same concatenated order, avoiding a temporary hard
predecessor tensor and a separate concatenation kernel.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _assemble(H, Q, TABLE, PREVIOUS, OUT, N:tl.constexpr, D:tl.constexpr,
              HS0:tl.constexpr, HS1:tl.constexpr, QS0:tl.constexpr, QS1:tl.constexpr,
              TS0:tl.constexpr, TS1:tl.constexpr, ORDER:tl.constexpr,
              B:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,B)
    token=tl.load(PREVIOUS+row).to(tl.int64)
    for branch in tl.static_range(len(ORDER)):
        if ORDER[branch]==0:
            value=tl.load(H+row*HS0+col*HS1,col<D,other=0)
        elif ORDER[branch]==1:
            value=tl.load(TABLE+token*TS0+col*TS1,col<D,other=0)
        else:
            value=tl.load(Q+row*QS0+col*QS1,col<D,other=0)
        tl.store(OUT+row*(len(ORDER)*D)+branch*D+col,value,col<D)


def assemble(head,hp,qp,previous):
    assert not head.training and not head.use_soft and head.use_hidden
    assert hp.dtype==qp.dtype==torch.bfloat16 and previous.is_contiguous()
    assert hp.shape==qp.shape and previous.shape==(hp.shape[0],)
    n,d=hp.shape
    table=head._previous_tables[n]
    assert table.shape[1]==d
    order=tuple({'hidden':0,'hard_previous':1,'q':2}[x] for x in head.branch_names)
    out=torch.empty((n,len(order)*d),device=hp.device,dtype=hp.dtype)
    _assemble[(n,)](hp,qp,table,previous,out,n,d,*hp.stride(),*qp.stride(),
                   *table.stride(),order,triton.next_power_of_2(d),num_warps=4)
    return out
