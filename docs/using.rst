Using ``pycle``
===============

.. _using pycle:

You can find a simple practical example of using pycle in this `jupyter notebook <notebooks/Demo_0-minimal_working_example.ipynb>`_. For more advanced and specific explanations, you can find the documentation in the related modules:

- Data generation and datasets: :mod:`pycle.utils.datasets`
- Normalization functions to prepare data for compressive learning: :mod:`pycle.utils.normalization`
- Metrics for evaluating performance: :mod:`pycle.utils.metrics`
- Storing and tracking results: :mod:`pycle.utils.intermediate_storage`
- Projectors to constrain the result: :mod:`pycle.utils.projectors`
- Sigma (sketching scale) estimation and mutualized sketching: :mod:`pycle.sketching.sigma_estimation` and see the `notebook tutorial <notebooks/sigma_estimation_tutorial.ipynb>`_
- Frequencies sampling utilities are available in :mod:`pycle.sketching.frequency_sampling`
- Feature maps are available in the module :mod:`pycle.sketching.feature_maps`. It is possible to build a custom feature map inheriting from the :class:`pycle.sketching.FeatureMap.FeatureMap` class. See on the module page.
- Solvers for compressive learning are available in the module :mod:`pycle.compressive_learning`. So far, only the solver for compressive Kmeans is available but it is possible to implement new ones. For solver aimed at learning a mixture, it is possible to inherit the base :class:`pycle.compressive_learning.SolverTorch.SolverTorch` class and stay in the framework. For solvers using CLOMP but for a different task, it is possible to inherit the base :class:`pycle.compressive_learning.CLOMP.CLOMP` class with all the CLOMP-related code already implemented but leaving room for specificities depending on the task.