# Guide to releasing a new lingvo pip package

Build the docker image for building the pip package

    docker build --tag tensorflow:lingvo_pip - < lingvo/pip_package/build.Dockerfile

Now, we assume that the repo is stored at /tmp/lingvo for the rest of these
instructions.

Enter the docker image, mapping /tmp/lingvo to /tmp/lingvo:

    docker run --rm -it -v /tmp/lingvo:/tmp/lingvo tensorflow:lingvo_pip bash

From the /tmp/lingvo directory, run

    rm -rf /tmp/lingvo_pip_package_build
    PYTHON_MINOR_VERSION=6 pip_package/build.sh
    PYTHON_MINOR_VERSION=7 pip_package/build.sh
    PYTHON_MINOR_VERSION=8 pip_package/build.sh

If everything goes well, this will produce a set of wheels in
/tmp/lingvo_pip_package_build.

    cd /tmp/lingvo_pip_pkg_build

To upload to the test pypi server:

    python3 -m twine upload --repository-url https://test.pypi.org/legacy/ *manylinux2010*.whl

To verify that it works as intended:

    python3 -m pip install -i https://test.pypi.org/simple/ --no-deps lingvo

You can test that the install worked for the common case by running a model
locally like:

    mkdir -p /tmp/lingvo_test/image
    cp -r /tmp/lingvo/lingvo/tasks/image/params/*.py /tmp/lingvo_test/image
    cd /tmp/lingvo_test
    python3 -m lingvo.trainer --model=image.mnist.LeNet5 --run_locally=cpu --logdir=/tmp/lenet5 --mode=sync

This should try to start training, but will fail if you haven't downloaded the
mnist dataset (see lingvo's base README.md).

If this works successfully, you can then upload to the production server as
follows.

    python3 -m twine upload *manylinux2010*.whl

And verify with:

    python3 -m pip install lingvo
