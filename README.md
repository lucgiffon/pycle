# The pycle toolbox
Pycle stands for PYthon Compressive LEarning; it is a toolbox of methods to perform compressive learning (also known as sketched learning), where the training set is not used directly but first heavily compressed. This learning paradigm allows:
    
- to learn from (very) large-scale datasets with drastically reduced computational resources;
- to learn on an untrusted machine without disclosing confidential data.

## Installation instructions

### Prerequisites
- g++ must be installed on your machine
- python version = 3.9
- install torch by following [these instructions](https://pytorch.org/get-started/locally/) according to your local configuration

To work with this repository, you must:

- create a clean python environment;
- clone this repository;
- go to the root folder and type:

```
pip install -r requirements.txt
```

To test that your installation works properly, you can do

```
pytest test -v
```

And make sure no test is FAILED. If one is, maybe try running the test suite again. If it persists, then raise an issue on github. It is normal behavior that you get a bunch of warnings.

## Build documentation

To build the documentation, you must have installed `pycle` first then go in the `$REPOSITORY_ROOT/docs/` directory and then type

```
make html
```

You can then view the documentation in your browser by opening `/docs/_build/html/index.html`.


## TODOS:
  
- UML diagram of the classes
- Quick demo video: overview of the code and possibilities (who is the target? what is the purpose?)
- Explanation of the projectors and how to implement new ones
- set the docs online
- Make sure vincent is ok with hosting the project on his github and giving me the maintainer right
- see with vincent to make it pip installable
- Check the demos
- Add tests to the notebooks (see nbmake online)
- Add usage example of all classes and functions in docstring
- Make the missing test see #cleaning todos
- differential privacy -> fix existing notebook (add functional tests)
- find coherence in numpy / torch dependence in functions (for instance: dataset module, )
- add "sep" method for sigma estimation (add functional tests)
- Maybe add emphasis somewhere on the OPU (feature maps? in using.rst)
- Estimate_sigma_by_entropy lacks docstring
- update guide.tex with most recent changes in architecture (or  remove it if there is redundancy with online doc).
- add doc to the vizualization module and functions
- should there be an "attribute" section in some classes docstring (CLOMP?)?
- make pytest ignore third party warnings i have no control over
- find a solution for all the maptlotlib windows opening during tests execution
- remove dependency to matplotlib and keep only pyplot figures
- Add support for GMM: test + exemple
- Add support for triple rademacher sketching op: test + exemple
- Add support for asymmetric feature map (rpf): test + exemple
- Add support of private sketching for the real variants of the considered maps
- Add the square nonlinearity, for sketching for PCA for example
- unbalanced case in generateCirclesDataset dataset function
- torchify somehow generateSpiralDataset (maybe also build-in the torch feature in all dataset generation procedures)

## Acknowledgement:

This toolbox is a fork from https://github.com/schellekensv/pycle where the original work have been produced. Some features of the original toolbox
have been lost in the process. 

This repository is essentially a port from `numpy` to `torch` of the original work. A lot of documentation have been added too.

