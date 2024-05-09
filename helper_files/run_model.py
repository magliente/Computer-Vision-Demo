from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from helper_files.model_and_cm import model_and_CM


def run_model(feature_desc, model_dict, label_dict, model_results, plot_cm=True, best_params = False, load_params = False):
  '''
  Run models on specified features and store results in `model_results` dictionary.

    Args:
        feature_desc -> dict: Dictionary containing feature sets for training and testing.
        model_dict -> dict: Dictionary containing model configurations.
        label_dict -> dict: Dictionary mapping label indices to label names.
        model_results -> dict: Dictionary to store model results.
        plot_cm (bool, optional): Whether to plot confusion matrices. Defaults to True.
        best_params (bool, optional): Whether to perform hyperparameter tuning using the best parameters. Defaults to False.
        load_params (bool, optional): Whether to load pre-defined parameters. Defaults to False.

    Returns:
        dict: Updated `model_results` dictionary containing model outcomes.
  '''
  
  # add run prediction labels
  for feature_name, features in feature_desc.items():
      for key, run in model_dict.items():

          # set training and comparison featuresets
          run['train'], run['train_labels'] = features['train']
          run['test'], run['test_labels'] = features['val']

          # grab
          run['label_dict'] = label_dict
          run['title'] = [f"Training - {feature_name}", f"Validation - {feature_name}"]

          if load_params:
            (model_results[feature_name][key]['model'],
            model_results[feature_name][key]['training_time'],
            model_results[feature_name][key]['preds'],
            model_results[feature_name][key]['inference_time']) = model_and_CM(name=key, **run, plot_cm=plot_cm, load_params = True, model_results = model_results)

          # if we are getting best parameters, extract the output correctly
          elif best_params:

            # Maintain class distribution across 5 kfold splits.
            run['cv'] = StratifiedKFold(n_splits=5)

            # Save results to dictionary
            (model_results[feature_name][key]['model'],
            model_results[feature_name][key]['training_time'],
            model_results[feature_name][key]['preds'],
            model_results[feature_name][key]['inference_time'],
            model_results[feature_name][key]['best_params']) = model_and_CM(name=key, **run, plot_cm=True, load_params = False)

          # if we are not getting best parameters, don't need to get value
          else:
            # Save results to dictionary
            (model_results[feature_name][key]['model'],
            model_results[feature_name][key]['training_time'],
            model_results[feature_name][key]['preds'],
            model_results[feature_name][key]['inference_time']) = model_and_CM(name=key, **run, plot_cm=plot_cm, load_params = False)
  return model_results
