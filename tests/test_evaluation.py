import sys
import os  # noqa
# os.chdir('..')
sys.path.insert(0, ".")  # noqa

import torch

torch.manual_seed(0)

import numpy as np
from utils.styled_plot import plt
from utils.wheat_seeds_dataset import Dataset, PyTorchDataset
from classifiers.mlp_classifier import MLPClassifier, WEIGHTS_FILE
from tests.config import WORKING_DIR

import matplotlib

matplotlib.use('agg')

module = __import__(f"{WORKING_DIR}.evaluation", fromlist=[
    'generate_explanations', 'get_most_important_features', 'get_predictions', 'compute_comprehensiveness', 'compute_sufficiency', 'remove_and_retrain', 'plot_scores'])


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


def test_generate_explanations():
    expl = module.generate_explanations(model, torch.ones(10, X_train.shape[1]))
    assert isinstance(expl, torch.Tensor)
    assert expl.shape == torch.Size([10, 7])
    assert expl.dtype == torch.float
    assert torch.allclose(expl[0], expl[1])
    assert torch.allclose(expl[0, 0], torch.tensor(0.5377), atol=1e-2)
    assert torch.allclose(expl[0, 2], torch.tensor(-0.028), atol=1e-2)

    expl = module.generate_explanations(model, X_test)
    assert isinstance(expl, torch.Tensor)
    assert expl.shape == X_test.shape
    assert torch.allclose(expl[0, 0], torch.tensor(0.006), atol=2e-3)
    assert torch.allclose(expl[0, -1], torch.tensor(-0.45), atol=2e-3)


def test_get_most_important_features():
    m = module.get_most_important_features(torch.tensor([[0.1, 0.2, 0.3]]))
    assert isinstance(m, torch.Tensor)
    assert m.shape == torch.Size([1, 1])
    assert m == torch.tensor([[2]])

    m = module.get_most_important_features(torch.tensor([[0.1, 0.2, 0.3]]), k=2)
    assert isinstance(m, torch.Tensor)
    assert m.shape == torch.Size([1, 2])
    assert torch.all(m == torch.tensor([[2, 1]]))

    m = module.get_most_important_features(torch.arange(0.0, 0.6, 0.01).reshape(6, 10), k=4)
    assert isinstance(m, torch.Tensor)
    assert m.shape == torch.Size([6, 4])
    assert torch.all(m[3] == torch.tensor([[9, 8, 7, 6]]))


def test_get_predictions():
    pred_labels, pred_probs = module.get_predictions(model, torch.arange(0.0, 0.7, 0.01).reshape(10, 7))
    assert isinstance(pred_labels, torch.Tensor)
    assert pred_labels.shape == torch.Size([10])
    assert torch.all(pred_labels[:5] == torch.tensor([0, 0, 0, 0, 1]))
    assert isinstance(pred_probs, torch.Tensor)
    assert pred_probs.shape == torch.Size([10, 3])
    assert torch.allclose(pred_probs[0], torch.tensor([0.7981, 0.1970, 0.0049]), atol=1e-2)

    feature_mask = torch.ones((7, 7), dtype=torch.long)
    feature_mask = feature_mask.scatter(1, torch.arange(0, 7).unsqueeze(1), 0)
    pred_labels, pred_probs = module.get_predictions(model, torch.arange(0.0, 0.7, 0.01)[:49].reshape(7, 7), feature_mask)
    assert isinstance(pred_labels, torch.Tensor)
    assert pred_labels.shape == torch.Size([7])
    assert torch.all(pred_labels[-3:] == torch.tensor([0, 1, 1]))
    assert isinstance(pred_probs, torch.Tensor)
    assert pred_probs.shape == torch.Size([7, 3])
    assert torch.allclose(pred_probs[-1], torch.tensor([0.2675, 0.587, 0.1455]), atol=1e-2)


def test_compute_comprehensiveness():
    scores, avg_score = module.compute_comprehensiveness(model, X_train)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == torch.Size([X_train.shape[0]])
    assert isinstance(avg_score, torch.Tensor)
    assert avg_score.shape == torch.Size([])

    assert torch.allclose(scores, torch.zeros(X_train.shape[0]).float())
    assert torch.allclose(avg_score, torch.tensor([0.0]))

    features_to_remove = torch.arange(0, 7).unsqueeze(0).repeat(X_train.shape[0], 1)
    scores, avg_score = module.compute_comprehensiveness(model, X_train, features_to_remove[:, :3])
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == torch.Size([X_train.shape[0]])
    assert isinstance(avg_score, torch.Tensor)
    assert avg_score.shape == torch.Size([])

    assert torch.allclose(scores[:4], torch.tensor([-0.2626, -0.0613, 0.5092, 0.4640]), atol=1e-2)
    assert torch.allclose(avg_score, torch.tensor([0.1665]), atol=1e-2)


def test_compute_sufficiency():
    scores, avg_score = module.compute_sufficiency(model, X_train)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == torch.Size([X_train.shape[0]])
    assert isinstance(avg_score, torch.Tensor)
    assert avg_score.shape == torch.Size([])

    assert torch.allclose(scores[:4], torch.tensor([-0.2295, 0.0598, 0.7632, 0.7796]), atol=1e-2)
    assert torch.allclose(avg_score, torch.tensor([0.3117]), atol=1e-2)

    features_to_keep = torch.arange(0, 7).unsqueeze(0).repeat(X_train.shape[0], 1)
    scores, avg_score = module.compute_sufficiency(model, X_train, features_to_keep[:, :3])
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == torch.Size([X_train.shape[0]])
    assert isinstance(avg_score, torch.Tensor)
    assert avg_score.shape == torch.Size([])

    assert torch.allclose(scores[:4], torch.tensor([0.0471, 0.1996, 0.5308, 0.5593]), atol=1e-2)
    assert torch.allclose(avg_score, torch.tensor([0.2032]), atol=1e-2)

    scores, avg_score = module.compute_sufficiency(model, X_train, features_to_keep)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == torch.Size([X_train.shape[0]])
    assert isinstance(avg_score, torch.Tensor)
    assert avg_score.shape == torch.Size([])

    assert torch.allclose(scores, torch.zeros(X_train.shape[0]).float())
    assert torch.allclose(avg_score, torch.tensor([0.0]))


def test_remove_and_retrain():
    y_train_t = torch.tensor(y_train).long()
    y_test_t = torch.tensor(y_test).long()
    expl_train = module.generate_explanations(model, X_train)
    expl_test = module.generate_explanations(model, X_test)
    new_model = MLPClassifier(input_units, output_units, 20)
    test_accuracy = module.remove_and_retrain(new_model, X_train.clone(), y_train_t, X_test.clone(), y_test_t, expl_train, expl_test, 2)

    assert isinstance(test_accuracy, torch.Tensor)
    assert test_accuracy.shape == torch.Size([])
    assert torch.allclose(test_accuracy, torch.tensor(0.8375), atol=3e-2)

    new_model = MLPClassifier(input_units, output_units, 20)
    test_accuracy = module.remove_and_retrain(new_model, X_train.clone(), y_train_t, X_test.clone(), y_test, expl_train, expl_test, 1)

    assert isinstance(test_accuracy, torch.Tensor)
    assert test_accuracy.shape == torch.Size([])
    assert torch.allclose(test_accuracy, torch.tensor(0.8750), atol=3e-2)


    new_model = MLPClassifier(input_units, output_units, 20)
    test_accuracy = module.remove_and_retrain(new_model, X_train.clone(), y_train_t, X_test.clone(), y_test, expl_train, expl_test, 6)

    assert isinstance(test_accuracy, torch.Tensor)
    assert test_accuracy.shape == torch.Size([])
    assert torch.allclose(test_accuracy, torch.tensor(0.4625), atol=3e-2)

    new_model = MLPClassifier(input_units, output_units, 20)
    test_accuracy = module.remove_and_retrain(new_model, X_train.clone(), y_train_t, X_test.clone(), y_test, expl_train, expl_test, 7)

    assert isinstance(test_accuracy, torch.Tensor)
    assert test_accuracy.shape == torch.Size([])
    assert torch.allclose(test_accuracy, torch.tensor(0.3250), atol=3e-2)


def test_plot_scores():
    module.plot_scores([1, 2, 3, 4], 'A title', 'x axis label', 'y axis label')
    fig = plt.gcf()
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == 'A title'
    assert ax.get_xlabel() == 'x axis label'
    assert ax.get_ylabel() == 'y axis label'
    assert np.all(ax.lines[0].get_ydata() == np.array([1, 2, 3, 4]))


if __name__ == "__main__":
    test_generate_explanations()
    test_get_most_important_features()
    test_get_predictions()
    test_compute_comprehensiveness()
    test_compute_sufficiency()
    test_remove_and_retrain()
