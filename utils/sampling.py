
import torch
import torch.nn.functional as F



def generalized_steps(model, x, time, numstep=10):
    with torch.no_grad():
        num_steps = numstep
        skip = time // num_steps
        epi = torch.randn_like(x[:, 0, :])
        seq = range(0, time, skip)
        n = epi.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [epi]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            xt = xs[-1].to('cuda')
            x0_t = model(xt, x, t)
            xs.append(x0_t[:, 0, :])
        final_prototype = xs[-1]


    return final_prototype
