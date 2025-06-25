from torch.optim import Adam, SGD, AdamW
from optims import AdaBound, SGDW
from optims import AdamW as AdamW_reimpl


def optimizer_interface(model, opt):
    if opt.optimizer == 'Adam':
        return Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    if opt.optimizer == 'SGD':
        return SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    if opt.optimizer == 'AdamW':
        return AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    if opt.optimizer == 'AdaBound':
        return AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.adabound_final_lr, weight_decay=opt.weight_decay)

    if opt.optimizer == 'SGDW':
        return SGDW(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    if opt.optimizer == 'AdamW_reimpl':
        return AdamW_reimpl(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    return None