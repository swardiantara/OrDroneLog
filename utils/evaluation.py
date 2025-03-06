import torch
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RegressionMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, preds, targets):
        """
        Store predictions and ground truth values.
        :param preds: torch.Tensor or np.array of shape (batch_size,)
        :param targets: torch.Tensor or np.array of shape (batch_size,)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        self.y_true.extend(targets)
        self.y_pred.extend(preds)

    def compute(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        kendall_tau, _ = kendalltau(y_true, y_pred)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "kendall": kendall_tau
        }
