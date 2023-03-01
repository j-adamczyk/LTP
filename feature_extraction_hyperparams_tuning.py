import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, cross_val_score
from tqdm import tqdm

from feature_extraction import calculate_features_matrix
from models import get_model


def perform_feature_extraction_hyperparams_tuning(
    features: pd.DataFrame,
    y: np.ndarray,
    model_type: str,
    verbose: bool = False,
) -> dict:
    ldp_params = ParameterGrid(
        {
            "n_bins": [30, 50, 70, 100],
            "normalization": ["none", "graph", "dataset"],
            "aggregation": ["histogram", "EDF"],
            "log_degree": [False, True],
        }
    )

    # we do not tune model hyperparameters during tuning feature extraction,
    # since that would take way too much time, and we don't tune RF anyway
    # since it's not really needed, so this tuning will be the only one that
    # runs in practice
    model = get_model(
        model_type=model_type,
        tune_model_hyperparams=False,
        verbose=verbose,
    )

    best_score = -np.inf
    best_params = None

    if verbose:
        print("Starting hyperparameter tuning")

    iterable = tqdm(ldp_params) if verbose else ldp_params

    for params in iterable:
        X = calculate_features_matrix(features, **params)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            n_jobs=-1,
        )
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_params = params

    if verbose:
        print("Best LDP hyperparameters:", best_params)

    return best_params
