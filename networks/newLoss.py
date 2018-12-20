import torch

def wingLoss(pred, target, w=10.0, epsilon=2.0):
    w, epsilon = torch.tensor([w]).cuda(), torch.tensor([epsilon]).cuda()
    dis = torch.abs_(pred - target)
    isSmall = dis <= w
    isLarge = dis > w
    small_loss = w * torch.log((isSmall * dis) / epsilon + 1)
    large_loss = isLarge * dis - w * (1 - torch.log(1 + w / epsilon))
    loss = small_loss + large_loss * isLarge
    loss = torch.mean(torch.sum(loss, dim=1),dim=0)
    return loss
