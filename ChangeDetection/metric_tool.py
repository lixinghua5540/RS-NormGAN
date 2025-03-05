import numpy as np


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False
    def get_comfuse_matrix(self):
        return self.confusion_matrix

###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        #print("n_class",n_class)

    def update_cm(self, pr, gt, weight=1):
        """acquire current confusion matrix and update the confusion matrix"""
        val = self.get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)

        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def get_confuse_matrix(self,num_classes, label_gts, label_preds):
    
        def __fast_hist(label_gt, label_pred):
            """
            Collect values for Confusion Matrix
            For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
            :param label_gt: <np.array> ground-truth
            :param label_pred: <np.array> prediction
            :return: <np.ndarray> values for confusion matrix
            """
            mask = (label_gt >= 0) & (label_gt < num_classes)
            hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],#
                               minlength=num_classes ** 2).reshape(num_classes, num_classes)
            return hist
    
        confusion_matrix = np.zeros((num_classes, num_classes))
        for lt, lp in zip(label_gts, label_preds):
            confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
        return confusion_matrix
def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    
    precision=hist.diagonal()/(np.sum(hist,axis=1)+np.finfo(np.float32).eps)
    recall=hist.diagonal()/(np.sum(hist,axis=0)+np.finfo(np.float32).eps)
    f1=np.mean(2*np.multiply(precision,recall)/(precision+recall+np.finfo(np.float32).eps))
    return f1

def cm2score(confusion_matrix):
    hist = confusion_matrix
    precision=hist.diagonal()/(np.sum(hist,axis=1)+np.finfo(np.float32).eps)
    recall=hist.diagonal()/(np.sum(hist,axis=0)+np.finfo(np.float32).eps)
    f1=np.mean(2*np.multiply(precision,recall)/(precision+recall+np.finfo(np.float32).eps))

    iou=hist.diagonal()/(np.sum(hist,axis=1)+np.sum(hist,axis=0)-hist.diagonal()+np.finfo(np.float32).eps)
    oa=np.sum(hist.diagonal())/np.sum(hist)
    col_sum=np.sum(hist, axis=1)
    raw_sum=np.sum(hist, axis=0)
    pe_fz = 0
    for i in range(6):
        pe_fz+=col_sum[i] * raw_sum[i]
    pe=pe_fz/(np.sum(hist)*np.sum(hist))
    kappa = (oa-pe) / (1+1e-15- pe)
    Recall=np.mean(recall)
    Precision=np.mean(precision)
    IoU=np.mean(iou)
    #score_dict = {'Kappa': kappa, 'IoU': iou, 'F1': f1, 'OA': oa, 'recall': recall, 'precision': precision, 'Pre': pre}
    score_dict = {'Kappa': kappa, 'F1': f1, 'OA': oa, 'recall': Recall, 'precision': Precision,'IoU': IoU}
    return score_dict



