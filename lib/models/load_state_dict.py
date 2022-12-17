try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def load_customized_state_dict(model, url, progress):
    state_dict = load_state_dict_from_url(url, progress=progress)
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(state_dict, strict=False)
    del state_dict