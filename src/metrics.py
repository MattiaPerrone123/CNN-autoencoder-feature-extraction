import torch
import scipy.stats as stats
import numpy as np


def get_iou_train(logits, targets):
    """Compute IoU between predicted masks and ground truth masks"""
    logits=torch.round(logits)
    targets=torch.round(targets)
    intersection=torch.logical_and(targets, logits).sum()
    union=torch.logical_or(targets, logits).sum()
    iou=intersection.item()/union.item()
    return iou


def get_mean_iou_train(inputs, outputs):
    """Compute the mean IoU across a batch of inputs and outputs"""
    batch_size=inputs.size(0)
    total_iou=0.0
    for i in range(batch_size):
        total_iou+=get_iou_train(inputs[i], outputs[i])
    mean_iou=total_iou/batch_size
    return mean_iou


def get_dice(logits, targets):
    """Compute the Dice score between predicted masks and ground truth masks"""
    true_positives=torch.logical_and(targets, logits).sum().item()
    false_positives=(logits-targets).clamp(min=0).sum().item()
    false_negatives=(targets-logits).clamp(min=0).sum().item()
    dice=(2.0*true_positives)/(2.0*true_positives+false_positives+false_negatives)
    return dice


def calculate_ci(data, confidence_level=0.95):
    """Calculate confidence intervals for the given data"""
    mean=np.mean(data)
    std=np.std(data, ddof=1)
    n=len(data)
    sem=std/np.sqrt(n)
    degrees_of_freedom=n-1
    ci_low, ci_high=stats.t.interval(confidence_level, df=degrees_of_freedom, loc=mean, scale=sem)
    return ci_low, ci_high


def combined_dice_bce_loss(pred, target, bce_weight=0.5, dice_weight=0.5, epsilon=1e-6):
    """Calculate the combined Dice and Binary Cross-Entropy (BCE) loss"""
    bce_loss=torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    pred_probs=torch.sigmoid(pred)
    pred_flat=pred_probs.view(-1)
    target_flat=target.view(-1)
    intersection=(pred_flat*target_flat).sum()
    dice_loss=1-(2.*intersection+epsilon)/(pred_flat.sum()+target_flat.sum()+epsilon)
    combined_loss=bce_weight*bce_loss+dice_weight*dice_loss
    return combined_loss


def calculate_iou_list(ground_truth, predictions, iou_function):
    """Calculate IoU for each pair of ground truth and prediction"""
    iou_list=[]
    for i in range(len(ground_truth)):
        iou_list.append(iou_function(ground_truth[i], torch.tensor(predictions[i])))
    return iou_list




