import argparse
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from sklearn.metrics import average_precision_score
import torch

def get_raw_dict(args):
    """
    return the dicf contained in args.
    
    e.g:
        >>> with open(path, 'w') as f:
                json.dump(get_raw_dict(args), f, indent=2)
    """
    if isinstance(args, argparse.Namespace): 
        return vars(args)   
    else:
        raise NotImplementedError("Unknown type {}".format(type(args)))

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def function_mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    average_precision_list = []

    for j in range(preds.shape[1]):
        average_precision_list.append(compute_avg_precision(targs[:, j], preds[:, j]))

    return 100.0 * float(np.mean(average_precision_list))


def compute_avg_precision(targs, preds):
    
    '''
    Compute average precision.
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''
    
    check_inputs(targs,preds)
    
    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = average_precision_score(targs, preds)
    
    return metric_value


def check_inputs(targs, preds):
    
    '''
    Helper function for input validation.
    '''
    
    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)