from .simmim import build_simmim


def build_model(args, is_pretrain=True):
    if is_pretrain:
        model = build_simmim(args)

    return model
