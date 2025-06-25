from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


def scheduler_interface(optimizer, opt):

    if opt.lr_scheduler == 'StepLR':
        return StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma)

    if opt.lr_scheduler == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones=opt.lr_milestones, gamma=opt.lr_gamma)

    if opt.lr_scheduler == 'ExponentialLR':
        return ExponentialLR(optimizer, gamma=opt.lr_gamma)

    if opt.lr_scheduler == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=opt.lr_T_max, eta_min=opt.lr_eta_min)

    if opt.lr_scheduler == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, mode=opt.lr_reduce_mode, factor=opt.lr_reduce_factor, patience=opt.lr_reduce_patience, \
                                threshold=opt.lr_reduce_threshold, cooldown=opt.lr_reduce_cooldown, min_lr=opt.lr_reduce_min_lr)

    return None