from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from pycle.utils import SingletonMeta


class IntermediateResultStorage(metaclass=SingletonMeta):
    def __init__(self):
        self.dct_objective_values = defaultdict(list)

    def add(self, elm, list_name):
        self.dct_objective_values[list_name].append(elm)

    def clear(self):
        self.dct_objective_values = defaultdict(list)

    def __getitem__(self, item):
        return self.dct_objective_values[item]

    def get_all_names(self):
        return sorted(list(self.dct_objective_values.keys()))

    def store_all_items(self, path_output_file):
        np.savez(path_output_file, **self.dct_objective_values)

    def load_items(self, path_input_file):
        z_loaded = np.load(path_input_file)
        self.dct_objective_values.update(
            **dict(z_loaded)
        )


class ObjectiveValuesStorage(IntermediateResultStorage):
    def get_objective_values(self, list_name):
        return self[list_name]

    def get_all_curve_names(self):
        return self.get_all_names()

    def store_objective_values(self, path_output_file):
        self.store_all_items(path_output_file)

    def load_objective_values(self, path_input_file):
        self.load_items(path_input_file)

    def show(self):
        fig, tpl_axs = plt.subplots(nrows=1, ncols=len(self.dct_objective_values))

        for idx_ax, (name_trace, lst_obj_values) in enumerate(self.dct_objective_values.items()):
            iter_ids = np.arange(len(lst_obj_values))
            objective_values = np.array(lst_obj_values)
            try:
                tpl_axs[idx_ax].plot(iter_ids, objective_values)
                tpl_axs[idx_ax].set_title(name_trace)
            except TypeError:
                assert len(self.dct_objective_values) == 1
                tpl_axs.plot(iter_ids, objective_values)
                tpl_axs.set_title(name_trace)

        plt.show()

