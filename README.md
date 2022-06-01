# The pycle toolbox
Pycle stands for PYthon Compressive LEarning; it is a toolbox of methods to perform compressive learning (also known as sketched learning), where the training set is not used directly but first heavily compressed. This learning paradigm allows:
    
- to learn from (very) large-scale datasets with drastically reduced computational resources;
- to learn on an untrusted machine without disclosing confidential data.

## Installation instruction

To work with this repository, you must:

- create a clean python environment; 
- install torch by following [these instructions](https://pytorch.org/get-started/locally/) according to your local configuration;
- clone this repository;
- go to the root folder and type:

```
pip install -r requirements.txt
```


## Contents of this repo:
For documentation:
* A "Guide" folder with semi-detailed introductory guide to this toolbox.
* A series of "DEMO_i" jupyter notebooks, that illustrate some core concepts of the toolbox in practice.

If you're new here, I suggest you start by opening either of those items first to get a hang of what this is all about.


The code itself, located in the "pycle" folder, structured into 3 main files:
* `sketching.py` contains everything related to building a feature map and sketching a dataset with it;
* `compressive_learning.py` contains the actual learning algorithms from that sketch, for k-means and GMM fitting for example;
* `utils.py` contains a diverse set of functions that can be useful, e.g., for generating synthetic datasets, or evaluating the learned models through different metrics and visualization utilities.

Note that if you want to use the core code of `pycle` direcltly without downloading this entire repository, you can install it directly from PyPI by typing
`pip install pycle`

## Todo documentation:

### Major
- Make sure installation is working from github
- add useful datasets to evaluate compressive learning (kddcup, breast-cancer, covtype) (dont forget to talk about normalization) (using.rst?)
- Explanation of the feature maps and how to implement new ones + make an example of custom feature map (using.rst?)
- Explanation of the frequencies sampling methods -> notebook presenting the methods (using.rst?)
- Explanation for the CLOMP decoders + make an example of custom solver (using.rst?)

  
### Minor
- UML diagram of the classes
- Quick demo video: overview of the code and possibilities (who is the target? what is the purpose?)
- Explanation of the projectors and how to implement new ones
- add Command to generate the doc in readme
- set the docs online
- Make sure vincent is ok with hosting the project on his github and giving me the maintainer right
- pip installable
- Check the demos
- differential privacy -> fix existing notebook (add functional tests)
- find coherence in numpy / torch dependence in functions
- add "sep" method for sigma estimation (add functional tests)
- Maybe add emphasis somewhere on the OPU (feature maps? in using.rst)

### TODOS FOR SKETCHING.PY

Short-term:
- Add utility function for the entropy based criterion
- Add support for GMM: test + exemple
- Add support for triple rademacher sketching op: test + exemple
- Add support for asymmetric feature map (rpf): test + exemple
- Add support of private sketching for the real variants of the considered maps
- Add the square nonlinearity, for sketching for PCA for example
- See with Titouan if the abstraction level of Solver is enough for his code.

Long-term:
- Fast sketch computation

## Citing this toolbox:
If you publish research using this toolbox, please follow this link for to get citation references (e.g., to generate BibTeX export files)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3855114.svg)](https://doi.org/10.5281/zenodo.3855114)

