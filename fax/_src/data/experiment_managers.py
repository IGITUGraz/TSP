import csv
import pathlib
from contextlib import ExitStack
from pathlib import Path

import numpy as np


class CSVManager:
    # simple scalar logger
    @staticmethod
    def mean_var(in_path: Path, out_path: Path, fields: list[str], delimiter:str = ","):
        filenames = [file for file in in_path.iterdir() if file.is_file()]
        data_rows = dict()
        data_mean = dict()
        data_var = dict()
        # return a safe stack of csv DictRfiles decriptor where first line is skip
        with ExitStack() as stack:
            list_fd_r = [csv.DictReader(stack.enter_context(open(fname, mode="r")),
                                        delimiter=delimiter) for fname in filenames]
            while list_fd_r[0]:
                try:
                    for i, fd_r in enumerate(list_fd_r):
                        row = next(fd_r)
                        for field in fields:
                            if data_rows.get(field) is None:
                                data_rows[field] = []
                            data_rows[field].append(row[field])
                except StopIteration:
                    break
        for field in fields:
            data = np.array(data_rows[field], dtype=np.float32)
            data = np.reshape(data, (-1, len(filenames)))
            data_mean[field] = data.mean(axis=1)
            data_var[field] = data.var(axis=1, ddof=1)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file_names = [out_path / f"{field}_mean_var.csv" for field in fields]
        for i, field in enumerate(fields):
            with open(out_file_names[i], mode="w") as fd_w:
                fd_w.writelines("id,data_mean,data_var\n")
                for j in range(len(data_mean[field])):
                    fd_w.writelines(f"{j+1},{str(data_mean[field][j])},{data_var[field][j]}\n")

    def __init__(self, path: pathlib.Path, name: str, delimiter: str = ","):
        self.store: dict[str, list] = dict()
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.path /= name
        self.delimiter = delimiter

    def append(self, key, value):
        entry = self.store.get(key)
        if entry is None:
            self.store[key] = []
        self.store[key].append(value)

    def done(self):
        keys = list(self.store)
        header = self.delimiter.join(keys)
        lines = [header + "\n"]
        for i in range(len(self.store[keys[0]])):
            new_line = ",".join([str(self.store[key][i]) for key in keys])
            lines.append(new_line + "\n")
        with open(self.path, mode="w") as fd_r:
            fd_r.writelines(lines)
