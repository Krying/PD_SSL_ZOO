from .data_pretrain import build_loader_simmim


def build_loader(args, is_pretrain):
    return build_loader_simmim(args)
    # if is_pretrain:
    #     return build_loader_simmim(args)
    # else:
    #     return build_loader_finetune(args)
