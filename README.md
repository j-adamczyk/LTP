# Local Topological Profile (LTP)

Code for paper "Strengthening topological baselines for graph classification using Local Topological Profile" J. Adamczyk, W. Czech. In this paper, we present Local Topological Profile (LTP), method optimizing and extending popular Local Degree Profile (LDP) baseline for topological graph classification.

To reproduce results from the paper, refer to `main.py` and function `perform_experiment()` in `perform_experiment.py`. Change values as needed. Datasets are downloaded automatically upon the first use.

Dependencies:
- Numpy
- NetworKit
- NetworkX
- Pandas
- PyTorch
- PyTorch Geometric
- Scikit-learn
- tqdm
- zstd

Optionally, to speed up estimators on Intel architectures: Intel(R) Extension for Scikit-learn
