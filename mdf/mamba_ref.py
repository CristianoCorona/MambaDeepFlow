"""Pure-PyTorch reference path for mamba_ssm.Mamba (mamba_impl="ref"; runs on CPU and GPU).

`selective_scan_ref` is adapted from mamba_ssm 2.3.2 (mamba_ssm/ops/selective_scan_interface.py,
Copyright (c) 2023 Tri Dao, Albert Gu, Apache-2.0), restricted to the real-valued, input-dependent B / C
case used by Mamba. It is copied rather than imported because that module imports the compiled
`selective_scan_cuda` extension at load time.

`mamba_forward_ref` repeats the non-fused path of `Mamba.forward` (mamba_ssm/modules/mamba_simple.py) with the
module's own parameters, using the PyTorch nn.Conv1d instead of causal_conv1d and the reference scan above.
Same parameters, so the state_dict is shared with the CUDA path.
"""

import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """u, delta, z: (b, d, l); A: (d, n); B, C: (b, n, l); D, delta_bias: (d,). Returns (b, d, l)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B, C = B.float(), C.float()
    x = A.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D[:, None]
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


def mamba_forward_ref(m, hidden_states):
    """Non-fused Mamba forward (training, no inference cache). hidden_states: (b, l, d_model)."""
    b, l, _ = hidden_states.shape
    xz = m.in_proj(hidden_states).transpose(1, 2)  # (b, 2*d_inner, l)
    A = -torch.exp(m.A_log.float())
    x, z = xz.chunk(2, dim=1)
    x = F.silu(m.conv1d(x)[..., :l])  # causal: padding d_conv-1 on both sides, keep the first l
    x_dbl = m.x_proj(x.transpose(1, 2).reshape(b * l, -1))
    dt, Bm, Cm = torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = (m.dt_proj.weight @ dt.t()).reshape(-1, b, l).transpose(0, 1)  # (b, d_inner, l)
    Bm = Bm.reshape(b, l, -1).transpose(1, 2).contiguous()
    Cm = Cm.reshape(b, l, -1).transpose(1, 2).contiguous()
    y = selective_scan_ref(x, dt, A, Bm, Cm, m.D.float(), z=z,
                           delta_bias=m.dt_proj.bias.float(), delta_softplus=True)
    return m.out_proj(y.transpose(1, 2))
