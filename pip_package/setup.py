# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution

__version__ = '0.6.3'
project_name = 'lingvo'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

REQUIRED_PACKAGES = [
    'ipykernel',
    'jupyter',
    'jupyter_http_over_ws',
    'matplotlib',
    'model-pruning-google-research',
    'Pillow',
    'protobuf>=3.8,<4',
    'sklearn',
    'sympy',
    'tensorflow-gpu>=2.0.0',
    'waymo-open-dataset-tf-2-0-0',
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
    packages=find_packages(include=['lingvo*'], exclude=[]),
    include_package_data=True,
    python_requires='>=3.6',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='Machine learning framework',
)
