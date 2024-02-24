# Generative Kernel Optimization (GKO)

GKO or Generative Kernel Optimization is a set of methods and tools for
optimizing programming kernels to achieve high performance with a
multi-stage method that includes performance evaluation, modeling, and
automated optimization. For more details on the algorithmic methods used
in GKO see below for a brief description and further reading sections.

## Surrogate Performance Modeling

Based on the performance measurements from the past runs, GKO builds a
performance surrogate model utilizing variants of Gaussian Process
Regression. The specific configurations for initial runs are based on
advanced sampling techniques that minimize the time spent in
bootstrapping the surrogate. This may sometimes become a multi-stage
processes that iteratively samples the performance configuration space,
builds a model, and subsequently evaluates the resulting uncertainty to
see if the stages have to be repeated to improve the precision of
predictions before moving to the optimization stage. Many of the stages
need advanced methods because of the high dimensionality and non-linear
nature of the optimization and loss hyper-surfaces.

## Installation

Please use the supplied `requirements.txt` to install the required
packages from PyPI. After dependent packages were installed, the
installation can be finished with `setuptools` using the provided
`pyproject.toml` file.

## Basic Usage

The simple invocation of the main entry point into the GKO methods
relies on the proper settings residing in the configuration files:

```sh
python3 -m gko [options]
```

## Publications (Further Reading)

The major methods used by GKO include sample-efficient algorithms for
sampling the configuration space that may have tens of dimensions, which
makes basic methods inefficient in both time and exploration
efficiency. The samples constitute a training set for building a
surrogate performance model that uses versions of Gaussian Process
Regression catered for the modeling circumstances. Subsequently, the
model used in the optimization stage to find optimal configuration of
parameters that can be used for obtaining high performance for the
sampled system and the application kernel combination.


```bibtex
@InProceedings{sidlakhdar2022dgp,
author={Sid-Lakhdar, Wissam M.  and Aznaveh, Mohsen and Luszczek, Piotr and Dongarra, Jack},
title={Deep {Gaussian} process with multitask and transfer learning for performance optimization},
booktitle={2022 IEEE High Performance Extreme Computing Conference (HPEC)},
year={2022},
publisher={IEEE},
pages={1-7},
isbn={1665497866}
}

@InProceedings{luszczek2022sabath,
author={Luszczek, Piotr and Brown, Cade},
title={Surrogate {ML/AI} Model Benchmarking for {FAIR} Principles' Conformance},
booktitle={2022 IEEE High Performance Extreme Computing Conference (HPEC)},
year={2022},
publisher={IEEE},
pages={1-5},
isbn={1665497866}
}

@article{luszczek2023dgpautotune
,author={Luszczek, Piotr and Sid-Lakhdar, Wissam M. and Dongarra, Jack}
,title={Combining multitask and transfer learning with deep {Gaussian} processes for autotuning-based performance engineering}
,journal={The International Journal of High Performance Computing Applications}
,year=2023
,note={DOI:10.1177/10943420231166365}
}
