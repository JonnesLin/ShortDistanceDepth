import numpy as np
import math
import torch


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    # return torch.log(x) / math.log(10)
    return np.log(x + 1e-10)


def evaluate_process(pred, gt):
    # Ignore 0
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    ratio = torch.median(gt) / torch.median(pred)
    pred *= ratio

    pred = pred.clip(min=1e-4, max=1.0)
    return pred, gt


def compute_metrics(pred, gt):
    n = pred.shape[0]
    output, target = evaluate_process(pred, gt)

    abs_diff = torch.abs(output - target)
    sq_diff = np.power(abs_diff, 2)
    mse = float(sq_diff.mean())
    rmse = math.sqrt(mse)
    lg10 = torch.pow(log10(output) - log10(target), 2).mean()
    log_rms = math.sqrt(lg10)
    absrel = float((abs_diff / target).mean())
    sqrel = float((sq_diff / target).mean())

    # calculate alpha1, alpha2, alpha3
    maxRatio = np.maximum(output.detach().numpy() / target.detach().numpy(),
                          target.detach().numpy() / output.detach().numpy())

    delta1 = float((maxRatio < 1.25).mean())
    delta2 = float((maxRatio < 1.25 ** 2).mean())
    delta3 = float((maxRatio < 1.25 ** 3).mean())

    inv_output = 1 / output
    inv_target = 1 / target
    abs_inv_diff = (inv_output - inv_target).abs()
    # RMS, log RMS, abs relative, sq relative
    return [rmse * n, log_rms * n, absrel * n, sqrel * n, delta1 * n, delta2 * n, delta3 * n]