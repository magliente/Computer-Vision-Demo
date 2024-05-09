
import time

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt


def model_and_CM(estimator, train, train_labels, test, test_labels, label_dict, name='', title=['',''],
               param_grid=None, cv=5, plot_cm=True, load_params = False, model_results = False):
    '''
    Train a specified estimator model and evaluate its performance using confusion matrix.

    Args:
        estimator: Sklearn-compatible estimator (e.g., LogisticRegression(), SVC(), RandomForest()).
        train -> array: Training data of shape (n_samples, n_features).
        train_labels -> array-like: Training labels of shape (n_samples,).
        test -> array: Testing data of shape (n_samples, n_features).
        test_labels -> array: Testing labels of shape (n_samples,).
        label_dict -> dict: Dictionary mapping label indices to label names.
        name (str, optional): Name of the model. Defaults to ''.
        title (list of str, optional): Titles for confusion matrix plots. Defaults to ['', ''].
        param_grid (dict, optional): Dictionary of parameters for hyperparameter tuning. Defaults to None.
        cv (int or custom_cv, optional): Number of cross-validation folds or custom CV object. Defaults to 5.
        plot_cm (bool, optional): Whether to plot confusion matrix. Defaults to True.
        load_params (bool, optional): Whether to load pre-defined parameters. Defaults to False.
        model_results: Nested dictionary with all model results used if we load in parameters.

    Returns:
        tuple: Tuple containing final trained model object, training time, prediction outputs, and inference time.

    '''
    if load_params:
      for feature, models in model_results.items():
        for model, run in models.items():
          if model == "Random Forest":
            pass
          else:
            best_params = run['best_params']
            

            # Filter parameters based on estimator type
            valid_params = {key: value for key, value in best_params.items() if key in estimator.get_params().keys()}

      # Set the parameters of the estimator
      model = estimator.set_params(**valid_params)

    elif param_grid is None:
        # model with default parameters
        model = estimator
    else:
        # optional parameter search, returns best search
        # custom_cv is essentially cross validation = 1 with dedicated train and validation set
        if cv == 'custom_cv':
            cv = custom_cv(train, test)
            search = GridSearchCV(estimator, param_grid=param_grid, scoring='accuracy', refit=True, n_jobs=-1, cv=cv)
            comb_features = np.concatenate((train, test))
            comb_labels = np.concatenate((train_labels, test_labels))
            search.fit(comb_features, comb_labels)
        else:
            search = GridSearchCV(estimator, param_grid=param_grid, scoring='accuracy', refit=True, n_jobs=-1, cv=cv)
            search.fit(train, train_labels)
        
        best_params = search.best_params_
        model = estimator.set_params(**best_params)

        print(
            "The best parameters are %s with a score of %0.2f"
            % (search.best_params_, search.best_score_)
        )
     
    # Train model with time keeping. Note parameter search will throw off
    start = time.monotonic()
    model.fit(train, train_labels)
    end = time.monotonic()
    print(f"{name} training time: {end - start:.2f} seconds")
    
    # make predictions and output confusion matrix
    to_pred = [(train, train_labels), (test, test_labels)]
    preds_output=[]
    for i, (data, labels) in enumerate(to_pred):
        # make predictions
        inf_start= time.monotonic()
        preds = model.predict(data)
        inf_end = time.monotonic()

        preds_output.append([preds])
        
        # Calculating the accuracy
        accuracy = accuracy_score(labels, preds)
        print(f"The {name} {title[i]} is {round(accuracy * 100, 2)}% accurate")
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds, normalize='pred')

        # display CM
        disp = ConfusionMatrixDisplay(cm, display_labels=label_dict.values())
        disp.plot(xticks_rotation=90)
        disp.ax_.set_title(f'{name}: {title[i]}\nAccuracy: {accuracy*100:.1f} percent', fontsize=14)
        plt.savefig(f"cm_plots/cm_{name}_{''.join(title[i])}", bbox_inches='tight')
        if not plot_cm:
            plt.close()

    preds_output_dict = {'model_train':preds_output[0], 'model_predict':preds_output[1]}

    
    if param_grid is None:
      return model, end-start, preds_output_dict, inf_end-inf_start
    else:
      return model, end-start, preds_output_dict, inf_end-inf_start, best_params


def custom_cv(train_features, val_features):
    '''
    Create custom cross validation indices for training and validation data.

    Parameters:
        train_features -> list: Training features.
        val_features -> list: Validation features.
    '''
    train_idc = list(range(len(train_features)))
    val_idc = list(range(len(train_features), len(train_features) + len(val_features)))
    return [(train_idc, val_idc)]

