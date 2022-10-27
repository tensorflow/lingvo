# Guide to releasing a new lingvo pip package

Update the version number in setup.py and commit it.

Build the docker image for building the pip package:

```sh
./pip_package/runner.sh
```

Then from within the image's environment, build the wheels:
```sh
./pip_package/invoke_build_per_interpreter.sh
```

If everything goes well, this will produce a set of wheels in
/tmp/lingvo/dist.

```sh
cd /tmp/lingvo/dist
```

To upload to the test pypi server:

```sh
python3.9 -m twine upload --repository-url https://test.pypi.org/legacy/ *manylinux2014*.whl
```

To verify that it works as intended:

```sh
python3 -m pip install -i https://test.pypi.org/simple/ --no-deps lingvo
```

You can test that the install worked for the common case by running a model
locally like:

```sh
mkdir -p /tmp/lingvo_test/image/params
cp -r /tmp/lingvo/lingvo/tasks/image/params/mnist.py /tmp/lingvo_test/image/params
cd /tmp/lingvo_test
python3 -m lingvo.trainer --model=image.mnist.LeNet5 --run_locally=cpu --logdir=/tmp/lenet5 --mode=sync
```

This should try to start training, but will fail if you haven't downloaded the
mnist dataset (see lingvo's base README.md).

If this works successfully, you can then upload to the production server as
follows.

```sh
cd /tmp/lingvo/dist
python3 -m twine upload *manylinux2014*.whl
```

And verify with:

```sh
python3 -m pip install lingvo
```

Remember to update the list of releases in the main README.
