import os
import sys
import warnings

from data_loading import DATASET_NAMES
from perform_experiment import perform_experiment

# the only warning raised is ConvergenceWarning for linear SVM, which is
# acceptable (max_iter is already higher than default); unfortunately, we
# have to do this globally for all warnings to affect child processes in
# cross-validation
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # also affect subprocesses

if __name__ == "__main__":
    n_bins = 50
    normalization = "graph"  # "none", "graph", "dataset"
    aggregation = "histogram"  # "histogram", "EDF"
    log_degree = False  # False, True

    # "LogisticRegression", "LinearSVM", "KernelSVM", "RandomForest"
    model_type = "RandomForest"
    for dataset_name in DATASET_NAMES:
        # TODO: check results
        print(dataset_name)
        perform_experiment(dataset_name)
