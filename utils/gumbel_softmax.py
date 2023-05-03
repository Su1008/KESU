import torch
import torch.nn.functional as F


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    if device.type != 'cpu':
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    #返回的结果是采样结果y_hard，而反向传播求导时却是对y求到的
    y_hard = (y_hard - y).detach() + y
    return y_hard


# test whether work or not
# a = torch.tensor([[0.56, 0.58, 0.67], [0.1, 0.9, 0.2]]).cuda()
# b = gumbel_softmax(a)
