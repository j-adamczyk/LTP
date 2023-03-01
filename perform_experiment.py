import gc

import numpy as np
from sklearn.metrics import accuracy_score

from caching import try_loading_cached_features, cache_features
from data_loading import load_dataset, load_dataset_splits
from feature_extraction import extract_features, calculate_features_matrix
from feature_extraction_hyperparams_tuning import (
    perform_feature_extraction_hyperparams_tuning,
)
from models import get_model


def perform_experiment(
    dataset_name: str,
    degree_sum: bool = False,
    shortest_paths: bool = False,
    edge_betweenness: bool = False,
    jaccard_index: bool = False,
    local_degree_score: bool = False,
    n_bins: int = 50,
    normalization: str = "none",
    aggregation: str = "histogram",
    log_degree: bool = False,
    model_type: str = "RandomForest",
    tune_feature_extraction_hyperparams: bool = False,
    tune_model_hyperparams: bool = False,
    use_features_cache: bool = True,
    verbose: bool = False,
):
    dataset = load_dataset(dataset_name)

    if use_features_cache:
        features = try_loading_cached_features(
            dataset_name,
            degree_sum=degree_sum,
            shortest_paths=shortest_paths,
            edge_betweenness=edge_betweenness,
            jaccard_index=jaccard_index,
            local_degree_score=local_degree_score,
        )
    else:
        features = None

    if not use_features_cache or features is None:
        features = extract_features(
            dataset,
            degree_sum=degree_sum,
            shortest_paths=shortest_paths,
            edge_betweenness=edge_betweenness,
            jaccard_index=jaccard_index,
            local_degree_score=local_degree_score,
            verbose=verbose,
        )

    if use_features_cache:
        cache_features(
            features,
            dataset_name=dataset_name,
            degree_sum=degree_sum,
            shortest_paths=shortest_paths,
            edge_betweenness=edge_betweenness,
            jaccard_index=jaccard_index,
            local_degree_score=local_degree_score,
        )

    y = np.array(dataset.data.y)

    # free memory - the original dataset will not be used anymore,
    # while it may double memory usage
    del dataset
    gc.collect()

    splits = load_dataset_splits(dataset_name)

    test_metrics = []
    for i, split in enumerate(splits):
        if verbose:
            print("Starting computing split", i)
        train_idxs = split.train_idxs
        test_idxs = split.test_idxs

        features_train = features.iloc[train_idxs, :]
        features_test = features.iloc[test_idxs, :]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        if tune_feature_extraction_hyperparams:
            ldp_params = perform_feature_extraction_hyperparams_tuning(
                features=features_train,
                y=y_train,
                model_type=model_type,
                verbose=verbose,
            )
        else:
            ldp_params = {
                "n_bins": n_bins,
                "normalization": normalization,
                "aggregation": aggregation,
                "log_degree": log_degree,
            }

        X_train = calculate_features_matrix(features_train, **ldp_params)

        X_test = calculate_features_matrix(
            features_test,
            **ldp_params,
        )

        model = get_model(
            model_type=model_type,
            tune_model_hyperparams=tune_model_hyperparams,
            verbose=verbose,
        )
        model.fit(X_train, y_train)
        if tune_model_hyperparams and verbose:
            try:
                best_params = model.best_params_
            except AttributeError:
                # custom handling for LogisticRegressionCV
                best_params = model.C_[0]
            print("Best hyperparams:", best_params)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_metrics.append(acc)

    acc_mean = np.mean(test_metrics)
    acc_stddev = np.std(test_metrics)

    return acc_mean, acc_stddev
