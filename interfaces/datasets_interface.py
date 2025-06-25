def dataset_interface(opt):
    aug_params = {
        'flip': opt.aug_flip,
        'rotate': opt.aug_rotate,
        'crop': opt.aug_crop,
        'crop_size': (opt.aug_crop_size, opt.aug_crop_size),
        'hsv': opt.aug_hsv,
        'hsv_limits': (opt.aug_hue_limit, opt.aug_sat_limit, opt.aug_val_limit),
        'shift_scale_rotate': opt.aug_shift_scale_rotate,
        'ssr_limits': (opt.aug_ssr_shift_limit, opt.aug_ssr_scale_limit, opt.aug_ssr_rotate_limit, opt.aug_ssr_aspect_limit),
        'brightness': opt.aug_brightness,
        'bright_limit': opt.aug_bright_limit,
        'noise': opt.aug_noise,
        'noise_mode': opt.aug_noise_mode,
        'blur': opt.aug_blur,
        'blur_mode': opt.aug_blur_mode,
        'blur_limit': opt.aug_blur_limit,
        'cutout': opt.cutout,
        # 'n_holes': opt.n_holes,
        # 'cut_size': opt.cut_size,
    }
    return dict(
        aug_params=aug_params
    )
