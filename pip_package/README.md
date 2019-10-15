# Guide to releasing a new lingvo pip package

Build the docker image for building the pip package

    docker build --tag tensorflow:lingvo_pip - < lingvo/pip_package/build.Dockerfile

Now, we assume that the repo is stored at /tmp/lingvo for the rest of these
instructions.

Enter the docker image, mapping /tmp/lingvo to /tmp/lingvo:

    docker run --rm -it -v /tmp/lingvo:/tmp/lingvo tensorflow:lingvo_pip bash


From the /tmp/lingvo directory, run

    bash pip_package/build.sh


If everything goes well, this will produce a set of wheels in /tmp/lingvo_pip_package_build.

    cd /tmp/lingvo_pip_pkg_build

To upload to the test pypi server:

    python3 -m twine upload --repository-url https://test.pypi.org/legacy/ *manylinux2010*.whl

To verify that it works as intended:

    pip3 install -i https://test.pypi.org/simple/ --no-deps lingvo

You can test that the install worked for the common case by running a model locally like:

    cd /tmp  # Move out of the lingvo source code dir.

    python3 -m lingvo.trainer --model=image.mnist.LeNet5 --run_locally=cpu --logdir=/tmp/lenet5 --mode=sync

This should try to start training, but will fail if you haven't downloaded the mnist dataset (see lingvo's base README.md).

If this works successfully, you can then upload to the production server as follows.

    python3 -m twine upload *manylinux2010*.whl

And verify with:

    pip3 install lingvo
