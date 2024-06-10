import os

import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize


descr = 'Stochastic algorithms for ICA'

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(lib_folder, "requirements.txt")
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

DISTNAME = 'mmica'
DESCRIPTION = descr
MAINTAINER = 'Pierre Ablin'
MAINTAINER_EMAIL = 'pierreablin@gmail.com'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/pierreablin/mmica.git'
URL = 'https://github.com/pierreablin/mmica'

extensions = [
    Extension(name="mmica._conjugate_gradient", sources=["mmica/_conjugate_gradient.pyx"])
]

setup(name='mmica',
      python_requires="<3.7",
      version="0.0.1",
      description=DESCRIPTION,
      long_description=open('README.md').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      install_requires=install_requires,
      packages=['mmica'],
      ext_modules=cythonize(extensions, include_path=[np.get_include()]),
      )
