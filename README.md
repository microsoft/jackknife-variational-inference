
# Jackknife Variational Inference, Python implementation

This repository contains code related to the following
[ICLR 2018](https://iclr.cc/Conferences/2018) paper:

* _Sebastian Nowozin_, "Debiasing Evidence Approximations: On
  Importance-weighted Autoencoders and Jackknife Variational Inference",
  [Forum](https://openreview.net/forum?id=HyZoi-WRb),
  [PDF](https://openreview.net/pdf?id=HyZoi-WRb).


## Citation

If you use this code or build upon it, please cite the following paper (BibTeX
format):

```
@InProceedings{
	title = "Debiasing Evidence Approximations: On Importance-weighted Autoencoders and Jackknife Variational Inference",
	author = "Sebastian Nowozin",
	booktitle = "International Conference on Learning Representations (ICLR 2018)",
	year = "2018"
}
```

## Installation

Install the required Python2 prequisites via running:

```
pip install -r requirements.txt
```

Currently this installs:

* [Chainer](http://chainer.org/), the deep learning framework, version 3.1.0
* [CuPy](http://cupy.chainer.org/), a CUDA linear algebra framework compatible
  with NumPy, version 2.1.0
* [NumPy](http://www.numpy.org/), numerical linear algebra for Python, version 1.11.0
* [SciPy](http://www.scipy.org/), scientific computing framework for Python, version 1.0.0
* [H5py](http://www.h5py.org/), an HDF5 interface for Python, version 2.6.0
* [docopt](http://docopt.org/), Pythonic command line arguments parser, version 0.6.2
* [PyYAML](https://github.com/yaml/pyyaml), Python library for
  [YAML](http://yaml.org) data language, version 3.12

## Running the MNIST experiment

To train the MNIST model from the paper, use the following parameters:

```
python ./train.py -g 0 -d mnist -e 1000 -b 2048 --opt adam \
    --vae-type jvi --vae-samples 8 --jvi-order 1 --nhidden 300 --nlatent 40 \
    -o modeloutput
```

Here the parameters are:

* `-g 0`: train on GPU device 0
* `-d mnist`: use the dynamically binarized MNIST data set
* `-e 1000`: train for 1000 epochs
* `-b 2048`: use a batch size of 2048 samples
* `--opt adam`: use the Adam optimizer
* `--vae-type jvi`: use _jackknife_ variational inference
* `--vae-samples 8`: use eight Monte Carlo samples
* `--jvi-order 1`: use first-order JVI bias correction
* `--nhidden 300`: in each hidden layer use 300 hidden neurons
* `--nlatent 40`: use 40 dimensions for the VAE latent variable

The training process creates a file `modeloutput.meta.yaml` containing the
training parameters as well as a directoy `modeloutput/` which contains a log
file and the serialized model which performed best on the validation set.

To evaluate the trained model on the test set, use

```
python ./evaluate.py -g 0 -d mnist -E iwae -s 256 modeloutput
```

This evaluates the model trained previously using the following test-time
evaluation setup:

* `-g 0`: use GPU device 0 for evaluation
* `-d mnist`: evaluate on the mnist data set
* `-E iwae`: use the IWAE objective for evaluation
* `-s 256`: use 256 Monte Carlo samples in the IWAE objective

Because test-time evaluation does not require backpropagation, we can evaluate
the IWAE and JVI objectives accurately using a large number of samples, e.g.
`-s 65536`.

The `evaluate.py` script also supports a `--reps 10` parameter which would
evaluate the same model ten times to investigate variance in the Monte Carlo
approximation to the evaluation objective.


## Choosing different objectives

As illustrated in the paper, the JVI objective generalizes both the ELBO and
the IWAE objectives.

For example, you can train on the importance-weighted autoencoder (IWAE)
objective using the parameter `--jvi-order 0` instead of `--jvi-order 1`.

You can train using the regular evidence lower bound (ELBO) by using the
special case of JVI, `--jvi-order 0 --vae-samples 1`, or directly via
`--vae-type vae`.

# Counting JVI sets

We include a small utility to count the number of subsets used by the
different JVI approximations.  There are two parameters, `n` and `order`,
where `n` is the number of samples of latent space variables per instance, and
`order` is the order of the JVI approximation (order zero corresponds to the
IWAE).

To run the utility, use:

```
python ./jvicount.py 16 2
```

This utility is useful because the set size can grow very rapidly for larger
JVI orders.  Therefore we can use the utility to assess the total number of
terms quickly and make informed choices about batch sizes and order of the
approximation.


# Contact

_Sebastian Nowozin_, `Sebastian.Nowozin@microsoft.com`


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

