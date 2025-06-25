import os
import time
from tqdm import tqdm


class Logger:

    def __init__(self, log_file, append=False):
        self.log_file = log_file
        self._check_file(self.log_file, not append)

    @classmethod
    def _check_file(cls, filename, reset=False):
        if os.path.isfile(filename):
            if reset:
                os.remove(filename)

    def log(self, log_type, message, show_time=True, print_type='print', not_to_screen=False, not_to_log=False):
        if log_type == 'None':
            log_info = ''
        else:
            log_info = '[{}] '.format(log_type)
        if show_time:
            time_info = time.strftime('[%y%m%d_%H:%M:%S]') + ' '
        else:
            time_info = ''
        text = '{}{}{}'.format(log_info, time_info, message)

        if not not_to_screen:
            self._print_to_screen(text, print_type)
        if not not_to_log:
            self._print_to_log(text)

    @classmethod
    def _print_to_screen(cls, text, print_type):
        if print_type == 'tqdm':
            tqdm.write(text)
        else:
            print(text)

    def _print_to_log(self, text):
        with open(self.log_file, 'a') as f:
            f.write(text + '\n')
