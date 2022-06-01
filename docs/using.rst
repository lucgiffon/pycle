Using ``pycle``
===============

Basic user
..........

.. _using pycle:

You can find a simple practical example of using pycle in this `jupyter notebook <notebooks/Demo_0-minimal_working_example.ipynb>`_. For more advanced and specific explanations, you can find the documentation in the related modules:

- Data generation and datasets: :mod:`pycle.utils.datasets` -> add useful datasets
- Metrics for evaluating performance: :mod:`pycle.utils.metrics`
- Storing and tracking results: :mod:`pycle.utils.intermediate_storage`
- Projectors to constrain the result: :mod:`pycle.utils.projectors`
- sigma estimation and mutualized sketching: :mod:`pycle.sketching.sigma_estimation` and see the `notebook tutorial <notebooks/sigma_estimation_tutorial.ipynb>`_
- feature maps :mod:`pycle.sketching.feature_maps` -> make an example of custom feature map
- solvers :mod:`pycle.compressive_learning` -> make an example of custom solver

Developers
..........
- torch functions :mod:`pycle.utils.torch_functions` -> explain why it is there and add the relevant information in the docstring
- encoding/decoding :mod:`pycle.utils.encoding_decoding` -> explain why it is there and add the relevant information in the docstring