Toolbox organization
....................

The ``pycle`` toolbox is a collection of several submodules:

- The ``sketching`` module instantiates feature maps then computes the sketch of datasets with it.
- The ``compressive_learning`` module contains the actual "learning" methods, extracting the desired parameters from the sketch.
- The ``utils`` module contains miscellaneous auxiliary functions, amongst others for generating synthetic datasets and evaluate the obtained solutions (as well quantitatively with well-defined metrics as qualitatively with visualization tools).
- The ``legacy`` module contains miscellaneous remains of the experimental process that led to the developpement of ``pycle``. It is here to serve as basis if some features must be re-habilitated.


Using Pycle
===========

Let's explore the core submodules of ``pycle``! Our focus here is understanding, so this section is a high-level tutorial rather than an exhaustive enumeration of what's inside the toolbox.

Sketching datasets
..................

To use the ``sketching`` submodule, you first need to import it; I often use \code{sk} as shorthand. Sketching, as defined in~\eqref{eq:sketching}, is done by simply calling ``sk.computeSketch``, as follows (we'll fill in the dots later):

.. code-block::

    import pycle.sketching as sk # import the sketching submodule

    X = ...    # load a numpy array of dimension (n,d)
    Phi = ... # sketch feature map, see later

    z = sk.computeSketch(X, Phi) # z is a numpy array containing the sketch


Learning from the sketch
........................

Advanced features of Pycle
==========================

Helpful tools from the utils.py submodule
.........................................

Designing the sampling pattern parameters when drawing the feature map
......................................................................

Sketching with differential privacy
...................................

Going further with pycle
========================

