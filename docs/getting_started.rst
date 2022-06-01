Getting Started
===============

What is compressive learning?
.............................

In usual machine learning, we fit some parameters :math:`\theta` (e.g., a parametric curve in regression, centroids in k-means clustering, weights in a neural network...) to a given set of training data :math:`X`. The actual algorithms for such tasks usually necessitate to access this training set multiple times (one complete pass on the dataset is sometimes called an "epoch"). This can be cumbersome when the dataset is extremely large, distributed across different machines and/or subject to privacy constraints. Compressive learning (CL, also called sketched learning) seeks to circumvent this issue by compressing the whole dataset into one lightweight "sketch" vector that requires only one pass on the dataset and can be computed in parallel. Learning is then done using only this (possibly rendered private) sketch instead of the inconveniently large dataset (that can be discarded). This allows to significantly reduce the computational resources that are needed to handle (very) massive collections of data.

.. _fig_compressive_learning:
.. figure:: /_static/images/compressive_learning.png
    :scale: 75 %
    :alt: The compressive learning workflow.
    :align: center

    The compressive learning workflow.

More precisely, CL comprises two steps (:numref:`fig_compressive_learning`):

**Sketching**:
    The dataset :math:`X = \{\bm{x}_i \: | \: i = 1, ..., n\}` (where we assume :math:`\bm{x}_i \in \mathbb R^d`) is compressed as a sketch vector that we note :math:`\bm{z}_X`, defined as the average over the dataset of some features :math:`\Phi(\bm{x}_i)` (the function :math:`\Phi : \mathbb R^d \mapsto \mathbb C^m` or :math:`\mathbb R^m` computes :math:`m` features, possibly complex):

    .. math::

        \label{eq:sketching}
        \bm{z}_X := \frac{1}{n} \sum_{i = 1}^n \Phi(\bm{x}_i).

    Since this is a simple averaging, sketching can be done on different chunks of :math:`X` independently, which is quite handy in distributed or streaming applications.

**Learning**:
    The target model parameters :math:`\theta` are then obtained by some algorithm :math:`\Delta` that operates *only* on this sketch,

    .. math::

        \theta = \Delta(\bm{z}_X).

    Typically, this involves solving some optimization problem :math:`\min_{\theta} f(\theta ; \bm{z}_X)`.

In the following, these steps are explained intuitively, the formal details being introduced only when needed; for a more solid/exhaustive overview of compressive learning, see `this tutorial paper <https://hal.inria.fr/hal-03350599/document>`_.

What to use ``pycle`` for ?
...........................

``pycle`` stands for "PYthon Compressive LEarning toolbox". It is a python library for machine learners and researchers to use and develop compressive learning algorithms.

-    Users will be able to:

        - Select a synthetic or real data dataset from :mod:`pycle.utils.datasets`;
        - Choose a feature map from a list of predefined ones in :mod:`pycle.sketching.feature_maps`;
        - Tune the feature map by choosing the random features distribution in :mod:`pycle.sketching.frequency_sampling` or by tuning the scale of the features in :mod:`pycle.sketching.sigma_estimation`;
        - Sketch a dataset with utility functions from :mod:`pycle.sketching`;
        - Learn from the sketch thanks to solvers in :mod:`pycle.compressive_learning`;
        - Evaluate the performance on the learning task with metrics in :mod:`pycle.utils.metrics`.

    Together, these modules containing functions and classes will allow any user to start toying with compressive learning.

-   Beyond standard use of ``pycle`` as a competing toolbox for their experiments, researchers can also extend the functionalities of ``pycle`` by:

        - Implementing other feature maps by subclassing the :mod:`pycle.sketching.feature_maps.FeatureMap.FeatureMap` base class;
        - Implementing other solvers for mixture model parameter estimation by subclassing the :mod:`pycle.compressive_learning.SolverTorch.SolverTorch` base class or even :mod:`pycle.compressive_learning.CLOMP.CLOMP` if the intent is rather to use CLOMP to solve an other task than compressive clustering.

    These base classes propose an interface through abstract methods which, if they are implemented in the child classes, can be used interchangeably.

Details regarding the other, more specific, features of pycle are available in :ref:`this page <using pycle>`.


Installation
............

Depending on wether you plan to simply *use* or rather *develop* in ``pycle``, you should choose one of two proposed installation methods. In both case, you need to first install ``pytorch`` by `following instruction on this page <https://pytorch.org/get-started/locally/>`_. Then you can procede to the installation of ``pycle``.

User installation:
    ``pycle`` is available in the Pypi repository and is installable through the simple command line:

    .. code-block::

        pip install pycle

Developper installation:
    If you plan on augmenting the functionalities of ``pycle``, you should rather clone the github repository and install ``pycle`` in editable mode:

    .. code-block::

        git clone https://github.com/schellekensv/pycle
        cd pycle
        pip install requirements.txt # installs pycle in editable mode and the requirements together



