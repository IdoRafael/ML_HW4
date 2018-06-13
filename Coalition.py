from DataPreparation import split_label
from ReadWrite import read_data


def load_prepared_data():
    return (read_data(x) for x in ['train.csv', 'validate.csv', 'test.csv'])


def test_model(model, name, parameters, train_x, train_y):
    score = make_scorer(f1_score, average='weighted')
    cv = 5

    print(name)
    start_time = time.time()
    classifier = GridSearchCV(
        model,
        parameters,
        cv=cv, scoring=score,
        n_jobs=-1
    ).fit(train_x, train_y)
    print(time.time() - start_time)
    return classifier


def run_experiments(train_x, train_y, names):
    # SVM
    svc = test_model(
        SVC(),
        'SVC',
        [{'kernel': ['rbf'], 'gamma': 10.0 ** np.arange(-9, 4, 1), 'C': 10.0 ** np.arange(-2, 11, 1)},
         {'kernel': ['linear'], 'C': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}],
        train_x, train_y
    )

    models = [svc, knn, tree, random_forest, gbc, mlp]
    save_models(models, names)

    return models

def optimize_models_parameters(train):
    train_x, train_y = split_label(train)
    names = ['SVC', 'KNN', 'DECISION_TREE', 'RANDOM_FOREST', 'GBC', 'MLP']
    models = run_experiments(train_x, train_y, names) if rerun_experiments else load_experiments(names)
    return models, names


def get_best_model(validate, models, names):
    pass


def predict_test_and_save_results(best_model, name, test):
    pass


def load_optimize_fit_select_and_predict():
    train, validate, test = load_prepared_data()

    models, names = optimize_models_parameters(train)

    best_model, name = get_best_model(validate, models, names)

    predict_test_and_save_results(best_model, name, test)