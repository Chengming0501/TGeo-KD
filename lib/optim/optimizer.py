import torch

def SGD(model, lr):
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        weight_decay=0)
    '''
    if cfg.model_arch == 'resnet152' or cfg.model_arch == 'resnet18' or cfg.model_arch == 'resnet50':
            ml.append({'params': model._modules['fc'].parameters(), 'lr': cfg.lr})
            for key, _ in (model._modules.items()):
                if (key != 'fc'):
                    ml.append({'params': model._modules[key].parameters()})
        else:
            ml.append({'params': model._modules['last_linear'].parameters(), 'lr': cfg.lr})
            for key, _ in (model._modules.items()):
                if (key != 'last_linear'):
                    ml.append({'params': model._modules[key].parameters()})
    '''

def Adam(model, lr):
    return torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)