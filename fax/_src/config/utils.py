import sys
import copy


def flatten_dict(d: dict, separator: str, exclude_list: list[str]):
    def flatten_dict_rec(sub_dict: dict, prefix: str, global_sub: dict):
        for k, v in sub_dict.items():
            if prefix + k not in exclude_list:
                if isinstance(v, dict):
                    flatten_dict_rec(v, prefix + k + separator, global_sub)
                else:
                    global_sub[prefix + k] = v

    flatten_d = dict()
    copy_d = copy.deepcopy(d)
    flatten_dict_rec(copy_d, "", flatten_d)
    return flatten_d


class OutputRedirect(object):
    def __init__(self, stdout, stderr) -> None:
        self.prev_stdout = sys.stdout
        self.prev_stderr = sys.stderr
        self.stderr_path = stderr
        self.stdout_path = stdout
        self.stdout_logger = None
        self.stderr_logger = None

    def __enter__(self):
        self.stdout_logger = Logger(self.stdout_path)
        self.stderr_logger = Logger(self.stderr_path)
        sys.stdout = self.stdout_logger
        sys.stderr = self.stderr_logger

    def __exit__(self, type, value, traceback):
        self.stdout_logger.close()
        self.stderr_logger.close()
        sys.stdout = self.prev_stdout
        sys.stderr = self.prev_stderr


class Logger(object):
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()
