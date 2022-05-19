"""
This module contains util classes for storing intermediate results and access it through a Singleton.

This is useful if one want to store, for instance, the intermediate step of an algorithm without having to
return it explicitely.
"""
from collections import defaultdict
from typing import NoReturn, Any

import numpy as np
from plotly.subplots import make_subplots

from pycle.utils import SingletonMeta
import plotly.graph_objects as go


class IntermediateResultStorage(metaclass=SingletonMeta):
    """
    This class can be called to store intermediate results identified by name.

    It contains a dictionary of lists in which the keys are the name of a given result and the
    lists contain all the results corresponding of that name.

    This class implements the Singleton Design pattern. It means that all instances of this class refer to the
    same object.

    Example:
        ```
        for i in range(n):
            intermediate_result = np.random.randn(d)
            IntermediateResultStorage().add(intermediate_result, "random_sample")
        ```

    """
    def __init__(self):
        self.dct_objective_values = defaultdict(list)

    def add(self, elm: Any, list_name: str) -> NoReturn:
        """
        Append the element `elm` to the list identified by `list_name`.

        Parameters
        ----------
        elm:
            The element to add. (for instance: an objective function value)
        list_name:
            The name of the list where to add the element.

        """
        self.dct_objective_values[list_name].append(elm)

    def clear(self) -> NoReturn:
        """
        Re_init the object to its initial state.
        """
        self.dct_objective_values = defaultdict(list)

    def __getitem__(self, item: str) -> list:
        """
        Parameters
        ----------
        item:
            The name of the list of values to get.

        Returns
        -------
            The list of values identified by `item`.
        """
        return self.dct_objective_values[item]

    def get_all_names(self) -> list:
        """
        Returns
        -------
            The list of all the list names sorted by alphabetical order.
        """
        return sorted(list(self.dct_objective_values.keys()))

    def store_all_items(self, path_output_file):
        """
        Stores all the lists in a npz file at the path `path_output_file`.

        Parameters
        ----------
        path_output_file:
            Path of the file storing all the lists.
        """
        np.savez(path_output_file, **self.dct_objective_values)

    def load_items(self, path_input_file) -> NoReturn:
        """
        Load all the lists from the given npz file.

        The elements are not returned but stored in the object.

        Parameters
        ----------
        path_input_file:
            A path to a npz file.
        """
        z_loaded = np.load(path_input_file)
        self.dct_objective_values.update(
            **dict(z_loaded)
        )


class ObjectiveValuesStorage(IntermediateResultStorage):
    """
    This class is just a proxy for the IntermediateResultStorage dedicated for storing objective function values.
    """
    def get_objective_values(self, list_name):
        return self[list_name]

    def get_all_curve_names(self):
        return self.get_all_names()

    def store_objective_values(self, path_output_file):
        self.store_all_items(path_output_file)

    def load_objective_values(self, path_input_file):
        self.load_items(path_input_file)

    def show(self, title="") -> NoReturn:
        """
        Show all the stored objective values in a Figure of many subplots
        in which each subplot correspond to one objective value.

        Parameters
        ----------
        title
            The main title of the figure.
        """
        max_number_of_columns = 3
        height_of_each_subplot = 400
        width_of_each_subplot = 600

        number_of_plots = len(self.dct_objective_values)
        number_of_rows = int(np.ceil(number_of_plots / max_number_of_columns))
        fig = make_subplots(rows=number_of_rows, cols=max_number_of_columns,
                            subplot_titles=list(self.dct_objective_values.keys()),
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        for idx_ax, (name_trace, lst_obj_values) in enumerate(self.dct_objective_values.items()):
            iter_ids = np.arange(len(lst_obj_values))
            objective_values = np.array(lst_obj_values)
            row_indice = int(idx_ax // max_number_of_columns + 1)
            col_indice = idx_ax % max_number_of_columns + 1
            fig.add_trace(
                go.Scatter(x=iter_ids, y=objective_values),
                row=row_indice, col=col_indice
            )

        fig.update_layout(height=height_of_each_subplot * number_of_rows,
                          width=width_of_each_subplot * max_number_of_columns,
                          margin=dict(l=5, r=5, t=100, b=5),
                          title_text=f"Objective values by iteration: {title}",
                          showlegend=False)
        fig.show()

