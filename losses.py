import torch
import torch.nn.functional as F

def one_hot_embedding(labels, num_classes=10, device=None):
    y = torch.eye(num_classes, device=device)
    return y[labels]

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(torch.cuda.current_device()) if use_cuda else "cpu")
    return device

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        ((alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha)))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, annealing_start=0.01, episode=0, increment=10, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_start = torch.tensor(annealing_start, dtype=torch.float32)
    annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / (annealing_step) * epoch_num)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        annealing_coef,
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1

    if episode == 0:
        if epoch_num == 0:
            return A
        kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
        return A*0.9 + kl_div*0.1
    else:
        kl_div = annealing_coef * kl_divergence(kl_alpha[:,-increment:], increment, device=device)
        return A*0.9 + kl_div*0.1

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, episode = 0, increment = 10, device=None):

    if not device:
        device = get_device()

    if episode == 0:
        alpha = torch.exp(output) + 1
    else:
        alpha = exp_evidence(output) + 1

    y = one_hot_embedding(target, num_classes, device).float()

    loss = torch.mean(
        edl_loss(
            torch.log, y, alpha, epoch_num, num_classes, annealing_step, episode=episode, increment=increment, device=device
        )
    )

    return loss
