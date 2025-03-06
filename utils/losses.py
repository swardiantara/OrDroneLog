import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from sentence_transformers.losses import ContrastiveLoss
from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer


def inverse_freq(y_train):
    class_counts = np.bincount(y_train)
    num_classes = len(class_counts)
    total_samples = len(y_train)

    class_weights = []
    for count in class_counts:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)
        
    return class_weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    

class SeverityFocal(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(SeverityFocal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        y_pred_probs = torch.softmax(inputs, axis=1)

        # Compute weights based on severity difference with higher penalty for false positives
        true_severity = targets.float()
        predicted_severity = y_pred_probs.argmax(dim=1).float()
        miss_penalty = torch.where(true_severity > predicted_severity,
                              2.5 * (1 + torch.abs(true_severity - predicted_severity)),
                              1 + torch.abs(true_severity - predicted_severity))
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss) * miss_penalty
        return loss.mean()
    

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute the BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        print(f"targets: {targets}")
        if targets[0] == 1:
            class_idx = 0
        else:
            if targets[2] == 1:
                class_idx = 2
            elif targets[1] == 1:
                class_idx = 1
            else:
                class_idx = 0
        # Compute the sigmoid and modulate it with the alpha and gamma factors
        p_t = torch.exp(-bce_loss)
        focal_loss = (self.alpha[class_idx] * (1 - p_t) ** self.gamma * bce_loss)
        
        return focal_loss.mean()
    

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, class_weights=None, reduction='mean'):
        """
        Multitask loss function for multi-label classification using BCE with class weights.

        Args:
        - alpha (float): Weight for the first task (binary classification).
        - beta (float): Weight for the second task (multi-label classification).
        - class_weights (Tensor, optional): Shape (4,), weights for each label.
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # self.class_weights = class_weights  # Per-class weights
        self.loss_fc = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction=reduction)
        self.loss_binary = None
        self.loss_multilabel = None

    def forward(self, predictions, targets):
        """
        Compute the multitask loss.

        Args:
        - predictions (Tensor): Shape (batch_size, 4), raw logits.
        - targets (Tensor): Shape (batch_size, 4), binary labels (0 or 1).

        Returns:
        - total_loss (Tensor): Weighted sum of losses.
        """
        # Compute BCE loss for all elements
        loss = self.loss_fc(predictions, targets)  # Shape: (batch_size, 4)

        # Apply task-specific weights
        loss_binary = loss[:, 0] # First element: binary classification
        loss_multilabel = loss[:, 1:] # Remaining elements: multi-label classification
        self.loss_binary = loss_binary
        self.loss_multilabel = loss_multilabel
        # Compute total weighted loss
        total_loss = self.alpha * loss_binary + self.beta * loss_multilabel
        return total_loss
    

class SeverityCE(nn.Module):
    def __init__(self, reduction='mean', class_weights=None):
        super(SeverityCE, self).__init__()
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, y_pred_logits, y_true):
        # Apply softmax to obtain class probabilities
        y_pred_probs = torch.softmax(y_pred_logits, axis=1)

        # Compute weights based on severity difference with higher penalty for false positives
        true_severity = y_true.float()
        predicted_severity = y_pred_probs.argmax(dim=1).float()
        weights = torch.where(true_severity > predicted_severity,
                              2.5 * (1 + torch.abs(true_severity - predicted_severity)),
                              1 + torch.abs(true_severity - predicted_severity))
        
        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(y_pred_logits, y_true, weight=self.class_weights) * weights

        # Apply reduction
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        # If reduction is None, return the raw loss tensor
        elif self.reduction is None:
            pass
        else:
            raise ValueError("Invalid reduction option. Please use 'mean', 'sum', or None.")

        return loss
    
    
# Define the Ordinal Cross-Entropy Loss
class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, weight=None):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits, targets):
        logits = logits.view(-1, self.num_classes)
        targets = targets.view(-1)

        # Compute the ordinal cross-entropy loss
        ordinal_loss = torch.tensor(0.0, requires_grad=True)
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                ordinal_loss += torch.sum(torch.clamp(logits[:, i] - logits[:, j], min=0) * (targets == j))

        return ordinal_loss
    
# taken from sentence-transformers
class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)

    """

    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        margin: float = 0.5,
        size_average: bool = True,
    ):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {"distance_metric": distance_metric_name, "margin": self.margin, "size_average": self.size_average}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
        )
        return losses.mean() if self.size_average else losses.sum()


class OrdinalContrastiveLoss(ContrastiveLoss):
    def __init__(
            self,
            model: SentenceTransformer,
            distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
            margin: float = 0.5,
            size_average: bool = True,
            num_classes: int = 4,
    ):
        super(OrdinalContrastiveLoss, self).__init__(model, distance_metric, margin, size_average)
        self.num_classes = num_classes

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], label_distances: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        # label_distances = label_distances / (self.num_classes - 1)
        margins = self.margin + label_distances # (2.0 - self.margin) *
        is_positive = (label_distances == 0).float()
        losses = 0.5 * (
            is_positive.float() * distances.pow(2) + 
            (1 - is_positive.float()) * F.relu(margins - distances).pow(2)
        )
        
        return losses.mean() if self.size_average else losses.sum()