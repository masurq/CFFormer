import torch as t

_name = 'ZMSegmentation_Plus'
_version = '1.5'
_author = 'SAR_Team_ZM'


def info():
    print('{} {}'.format(_name, _version))
    print('Author: {}'.format(_author))
    print('PyTorch Version {}, {} Version {}'.format(t.__version__, _name, _version))


def info_logger(logger):
    logger.log('None', '{} {}'.format(_name, _version), show_time=False, print_type='print')
    logger.log('None', 'Author: {}'.format(_author), show_time=False, print_type='print')
    logger.log('None', 'PyTorch Version {}, {} Version {}'.format(t.__version__, _name, _version), show_time=False,
               print_type='print')
