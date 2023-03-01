from sklearn.model_selection import GridSearchCV

# use Intel Extensions for Scikit-learn, if they are available
try:
    from sklearnex.ensemble import RandomForestClassifier
    from sklearnex.svm import SVC
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

from sklearn.svm import LinearSVC


def get_model(
    model_type: str,
    tune_model_hyperparams: bool = False,
    verbose: bool = False,
):
    if model_type == "LinearSVM":
        model = LinearSVC(
            penalty="l2",
            loss="hinge",
            max_iter=10000,
            random_state=0,
        )
        params_grid = {
            "C": [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
        }
    elif model_type == "KernelSVM":
        model = SVC(
            kernel="rbf",
            cache_size=1024,
            random_state=0,
        )
        params_grid = {
            "C": [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
            "gamma": [1e-2, 1e-1, 1, 1e1, 1e2],
        }
    elif model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=500,
            criterion="gini",
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        )
        # we do not perform hyperparameter tuning for RF
        params_grid = {}
    else:
        raise ValueError(f"Model type '{model_type}' not supported")

    if tune_model_hyperparams:
        # GridSearchCV has weird verbosity settings, to get reasonably verbose outputs
        # we need to set 2
        verbose = 2 if verbose else 0

        return GridSearchCV(
            estimator=model,
            param_grid=params_grid,
            n_jobs=-1,
            cv=5,
            verbose=verbose,
        )
    else:
        return model
