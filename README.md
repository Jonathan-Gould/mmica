# Majorization-minimization ICA
This repository contains the code for the AISTATS 2019 paper "Stochastic algorithms with descent guarantees for ICA":

> Ablin, P., Gramfort, A., Cardoso, J.F. & Bach, F. (2019). Stochastic algorithms with descent guarantees for ICA. Proceedings of Machine Learning Research, in PMLR 89:1564-1573

### Installation
To get started, clone the repository, activate a python 3.6 virtual environment, install the project's requirements, and install the package.

```shell
# create and activate a virtual environment with python 3.6
conda create -y -n mmica python=3.6 && conda activate mmica

# install the packages in requirements.txt
# (Note that while setuptools can technically do this step, sometimes the build fails.)
conda install --yes -c conda-forge --file requirements.txt 

# compile the extension and install mmica as a package
python setup.py install

# sometimes this helps when the build fails:
# python setup.py build_ext --inplace 
```

### API

There are two solvers in the package:

* `solver_incremental` takes a `(p, n)` array as input
* `solver_online` takes a generator as input

### Examples
Incremental solver:
```python
import numpy as np
from mmica import solver_incremental

p, n = 2, 1000

S = np.random.laplace(size=(p, n))
A = np.random.randn(p, p)
X = A.dot(S)

W = solver_incremental(X)
print(np.dot(W, A))  # close from a permutation + scale matrix
```

Online solver:

```python
import numpy as np
from mmica import solver_online

p = 2
batch_size = 100

A = np.random.randn(p, p)
S = (np.random.laplace(size=(p, batch_size)) for _ in range(20))
X = (A.dot(s) for s in S)
W = solver_online(X, p)

print(np.dot(W, A))  # close from a permutation + scale matrix
```
