import torch


def get_is_cuda():
    return True if torch.cuda.is_available() else False


def get_tensors_type():
    return torch.cuda.FloatTensor if get_is_cuda() else torch.FloatTensor