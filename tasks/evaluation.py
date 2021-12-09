
import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
from utils.styled_plot import plt
from utils.wheat_seeds_dataset import Dataset, PyTorchDataset
from classifiers.mlp_classifier import MLPClassifier, WEIGHTS_FILE
from captum.attr import Saliency


torch.manual_seed(0)


def generate_explanations(model, X):
    """
    Generates explanations for the given model for each instance in X.
    Uses captum.attr.Saliency to generate exlanations based on the input gradient method as the explainer.

    Params:
        model: (MLPClassifier, torch.nn.Module): A classification model. Has a .predict method that returns a tuple (predicted label, probability scores for each class).
        X (torch.Tensor): The dataset to generate explanations for, of shape (num. instances, num. features)
    Returns:
        explanations (torch.Tensor): The explanations of shape (num. instances, num. features).

    Hint: set abs=False in Saliency() to avoid taking the absolute gradients.
    """
    return None


def get_most_important_features(explanations, k=1):
    """
    Given explanations, returns the indices of the k most important feature for each instance.
    Params:
        explanations (torch.Tensor): The explanations of shape (num. instances, num. features).
        k (int): How many features to consider.
    Returns:
        important_feature_indices (torch.tensor):
            The indices of the most important features for each instance, of shape (num. instances, k).
            For example, if the explanations for instance [0] are [0.1, 0.2, 0.4, 0.3] and k=2, then important_feature_indices[0] = [2, 3]
    """
    return None

def get_predictions(model, X, feature_mask=None):
    """
    Runs a given model to make predictions on a given dataset. Optionally applies a feature mask before predicting.
    Params:
        model: (MLPClassifier, torch.nn.Module): A classification model. Has a .predict method that returns a tuple (predicted label, probability scores for each class).
        X (torch.Tensor): The dataset to predict, of shape (num. instances, num. features)
        feature_mask (optional, torch.Tensor): a binary mask of the same shape as X. A zero denotes that the corresponding element in X is to be replaced with a zero.
    Returns:
        pred_labels (torch.Tensor): The predicted labels, of shape (num. instances)
        pred_probabilities (torch.Tensor): The predicted probabilities for each class, of shape (num. instances, num. classes)
    """
    return None


def compute_comprehensiveness(model, X, features_to_remove=None):
    """
    Computes the comprehensiveness metric, by measuring the difference in output probability of the predicted class between the original prediction and the prediction with some features removed.
    Params:
        model: (MLPClassifier, torch.nn.Module): A classification model. Has a .predict method that returns a tuple (predicted label, probability scores for each class).
        X (torch.Tensor): The dataset to compute the comprehensiveness, of shape (num. instances, num. features)
        features_to_remove (optional, torch.Tensor): A tensor of shape (num. instances, k), containing the features indices of which features to remove for each instance.
    Returns:
        comprehensiveness_scores (torch.Tensor): The comprehensiveness score of each instance, of shape (num. instances)
        avg_comprehensiveness(torch.Tensor): The average comprehensiveness score across the instances.
    """
    return None


def compute_sufficiency(model, X, features_to_keep=None):
    """
    Computes the sufficiency metric, by measuring the difference in output probability of the predicted class between the original prediction and the prediction with only some features kept.
    Params:
        model: (MLPClassifier, torch.nn.Module): A classification model. Has a .predict method that returns a tuple (predicted label, probability scores for each class).
        X (torch.Tensor): The dataset to compute the sufficiency, of shape (num. instances, num. features)
        features_to_keep (optional, torch.Tensor): A tensor of shape (num. instances, k), containing the features indices of which features to keep for each instance.
    Returns:
        sufficiency_scores (torch.Tensor): The sufficiency score of each instance, of shape (num. instances)
        avg_sufficiency(torch.Tensor): The average sufficiency score across the instances.
    """
    return None


def remove_and_retrain(model, X_train, y_train, X_test, y_test, expl_train, expl_test, k):
    """
    Removes the k most important features from a dataset (per instance) and retrains a model to obtain a new test accuracy.
    The removed features are replaced with a baseline value, that is randomly drawn from a uniform distribution in the interval [0, 1).
    Params:
        model: (MLPClassifier, torch.nn.Module): A new, untrained classification model. Has a .fit method that expects a trainining and test dataset. You can create these using PyTorchDataset. For example, you can create the training set using PyTorchDataset(X_train, y_train).
        X_train (torch.Tensor): The train dataset of shape(num. instances, num. features).
        y_train (torch.Tensor): The labels for the train dataset, of shape (num. instances).
        X_test (torch.Tensor): The test dataset of shape(num. instances, num. features).
        y_test (torch.Tensor): The labels for the test dataset, of shape (num. instances).
        expl_train (torch.Tensor): The explanations generated using X_train, of same shape as X_train.
        expl_test (torch.Tensor): The explanations generated using X_test, of same shape as X_test.
        k (int): The number of most important features to remove per instance.
    Returns:
        test_accuracy (torch.Tensor): The test accuracy obtained by fitting the new model.

    Hint: Use torch.rand to draw baseline values.
    """
    return None


def plot_scores(scores, title, xlabel, ylabel):
    """
    Plots a list of scores in a line chart, using the provided title and axis labels.
    Params:
        scores (List[float]): The scores to plot on the y-axis.
        title (str): The title to use for the plot.
        xlabel (str): The label to use for the x-axis of the plot.
        ylabel (str): The label to use for y-axis of the plot.
    
    Hint: Do not call plt.show() here.
    """



if __name__ == "__main__":
    dataset = Dataset(
        "wheat_seeds",
        [0, 1, 2, 3, 4, 5, 6],
        [7],
        normalize=True,
        categorical=True)

    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    input_units = X_train.shape[1]
    output_units = len(dataset.get_classes())
    features = dataset.get_input_labels()

    ds_train = PyTorchDataset(X_train, y_train)
    ds_test = PyTorchDataset(X_test, y_test)

    model = MLPClassifier(input_units, output_units, 20)

    if WEIGHTS_FILE.exists():
        model.load_state_dict(torch.load(WEIGHTS_FILE))
    else:
        model.fit(ds_train, ds_test)
        torch.save(model.state_dict(), WEIGHTS_FILE)

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()

    print('Running `generate_explanations`...')
    explanations = generate_explanations(model, X_train)

    print('Running `get_most_important_features`...')
    most_important_features = get_most_important_features(explanations, k=7)

    print('Running `compute_comprehensiveness`...')
    comps = [compute_comprehensiveness(model, X_train, most_important_features[:, :k])[1] for k in range(1, 7+1)]
    comps = [compute_comprehensiveness(model, X_train)[1]] + comps
    print(f'Avg. Comprehensiveness when removing')
    for k in range(0, 7):
        print(f'{k+1} features: {comps[k]:.3f}')

    print('Running `compute_sufficiency`...')
    suffs = [compute_sufficiency(model, X_train, most_important_features[:, :k])[1].item() for k in range(1, 7+1)]
    suffs = [compute_sufficiency(model, X_train)[1]] + suffs
    print(f'Avg. sufficiency when including')
    for k in range(0, 7):
        print(f'{k+1} features: {suffs[k]:.3f}')

    print('Running `remove_and_retrain`...')
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()
    expl_train = explanations
    expl_test = generate_explanations(model, X_test)
    test_accuracies = []
    for k in range(0, 7+1):
        new_model = MLPClassifier(input_units, output_units, 20)
        test_accuracies.append(
            remove_and_retrain(
                new_model,
                X_train.clone(),
                y_train,
                X_test.clone(),
                y_test,
                expl_train,
                expl_test,
                k
            ).item()
        )
    print(f'Test Accuracy when removing')
    for num_removed, acc in zip(range(0, 7+1), test_accuracies):
        print(f'\t{num_removed} features: {acc:.2f}')

    print('Plotting comprehensiveness scores...')
    plot_scores(comps, 'Comprehensiveness', 'num. features removed', 'avg. difference to original prediction')
    plt.show()

    print('Plotting sufficiency scores...')
    plot_scores(suffs, 'Sufficiency', 'num. features included', 'avg. difference to original prediction')
    plt.show()

    print('plotting Remove and Retrain scores')
    plot_scores(test_accuracies, 'Remove and Retrain', 'num. features removed', 'test accuracy')
    plt.show()
