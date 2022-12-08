# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup script for pip package."""
import sys

from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution

__version__ = '0.12.6'
project_name = 'lingvo'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

REQUIRED_PACKAGES = [
    'attrs',
    'etils',
    'graph-compression-google-research',
    'ipykernel',
    'jupyter_http_over_ws',
    'jupyter',
    'matplotlib',
    'model-pruning-google-research',
    'Pillow',
    'protobuf',
    'scikit-learn',
    'sentencepiece',
    'sympy',
    'tensorflow-datasets',
    'tensorflow-hub',
    'tensorflow-text~=2.9.0',
    'tensorflow~=2.9.2',
]


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


class InstallCommand(install):
  """Override install command.

  Following:
  https://github.com/bigartm/bigartm/issues/840.
  """

  def finalize_options(self):
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib


setup(
    name=project_name,
    version=__version__,
    description=('Lingvo libraries.'),
    author='Lingvo Authors',
    author_email='lingvo-bot@google.com',
    packages=find_namespace_packages(
        include=find_namespace_packages(
            include=['lingvo*'], exclude=['*.params*'])),
    include_package_data=True,
    python_requires='>=3.8,<3.11',
    install_requires=REQUIRED_PACKAGES,
    zip_safe=False,
    cmdclass={
        'install': InstallCommand,
    },
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='Machine learning framework',
)
