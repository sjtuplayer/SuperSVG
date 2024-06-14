import torch
import numpy as np


def euclidean_dist_func(x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    return torch.pow(x - y, 2).sum(3)


def loss_dpw(strokes, strokes_b, gamma=0.01, bandwidth=100): #strokes:generated paths; strokes_b:target paths
    d_xy = euclidean_dist_func(strokes, strokes_b)
    d_xy.retain_grad()
    p = torch.ones((d_xy.shape[0], d_xy.shape[1] + 2, d_xy.shape[2] + 2), requires_grad=True)
    p = p * torch.inf
    p[:, 0, :] = 0
    q = torch.ones((d_xy.shape[0], d_xy.shape[1] + 2, d_xy.shape[2] + 2), requires_grad=True)
    q = q * torch.inf
    p.retain_grad()
    q.retain_grad()
    result = torch.ones((d_xy.shape[0]), requires_grad=True)
    result = result * torch.inf
    result.retain_grad()

    B = d_xy.shape[0]
    N = d_xy.shape[1]
    M = d_xy.shape[2]
    P = p.clone()
    Q = q.clone()
    for b in range(B):
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue
                r0 = -Q[b, i - 1, j] / gamma
                r1 = -P[b, i - 1, j] / gamma
                r2 = -Q[b, i, j - 1] / gamma
                r3 = -P[b, i, j - 1] / gamma
                rmax = max(r0, r1)
                rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax)
                softmin = -gamma * (torch.log(rsum) + rmax)
                P[b, i, j] = d_xy[b, i - 1, j - 1] + softmin
                rmax0 = max(r2, r3)
                rsum0 = torch.exp(r2 - rmax0) + torch.exp(r3 - rmax0)
                softmin0 = -gamma * (torch.log(rsum0) + rmax0)
                Q[b, i, j] = softmin0
                if r2 == -torch.inf and r3 == -torch.inf:
                    Q[b, i, j] = torch.inf
        p0 = -Q[b, N, M] / gamma
        q0 = -P[b, N, M] / gamma
        rmax1 = max(p0, q0)
        rsum1 = torch.exp(p0 - rmax1) + torch.exp(q0 - rmax1)
        softmin1 = -gamma * (torch.log(rsum1) + rmax1)
        result[b] = softmin1

    return result.mean()
