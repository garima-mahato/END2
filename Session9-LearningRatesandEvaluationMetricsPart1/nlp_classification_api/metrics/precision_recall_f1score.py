import numpy as np

class precision_recall_f1score:
    def __init__(self, labels=None):
        # if labels is not None:
        #     assert isinstance(labels, list), 'labels must be list of integers starting with 0'
        self.labels = labels
        self.classwise_correct_pred = {c: 0 for c in self.labels}
        self.classwise_pred = {c: 0 for c in self.labels}
        self.classwise_target = {c: 0 for c in self.labels}
        self.classwise_prec_recall_f1 = {c: {'precision': 0, 'recall': 0, 'f1': 0} for c in self.labels}
        self.global_tp = 0
        self.global_fp = 0
        self.global_fn = 0
        self.micro_avg_f1, self.micro_prec, self.micro_recall, self.macro_f1, self.macro_precision, self.macro_recall, self.wgt_f1, self.wgt_precision, self.wgt_recall = 0,0,0,0,0,0,0,0,0

    def update(self, preds, targets):
        try:
            # if len(targets.shape)!=2:
            #     targets = targets.unsqueeze(1)
            top_pred = preds.argmax(1, keepdim = True)
            self.global_tp += top_pred.eq(targets.view_as(top_pred)).sum().item()
            for c in self.classwise_correct_pred.keys():
                self.classwise_correct_pred[c] += (top_pred[top_pred.eq(targets.view_as(top_pred))]==c).sum().float()
                self.classwise_pred[c] += (top_pred==c).sum().float()
                self.classwise_target[c] += (targets==c).sum().float()
        except Exception as e:
            raise e
    
    def calculate(self):
        try:
            for k in self.labels:
                self.classwise_prec_recall_f1[k]['precision'] = (self.classwise_correct_pred[k]/self.classwise_pred[k]).item()
                self.classwise_prec_recall_f1[k]['recall'] = (self.classwise_correct_pred[k]/self.classwise_target[k]).item()
                self.classwise_prec_recall_f1[k]['f1'] = (self.classwise_prec_recall_f1[k]['precision'] * self.classwise_prec_recall_f1[k]['recall'])/(self.classwise_prec_recall_f1[k]['precision'] + self.classwise_prec_recall_f1[k]['recall'] + 1e-20)

                self.global_fp += self.classwise_pred[k] - self.classwise_correct_pred[k]
                self.global_fn += self.classwise_target[k] - self.classwise_correct_pred[k]

            self.micro_prec = self.global_tp/(self.global_tp+self.global_fp.item())
            self.micro_recall = self.global_tp/(self.global_tp+self.global_fn.item())
            self.micro_avg_f1 = (self.micro_prec * self.micro_recall) / (self.micro_prec + self.micro_recall + 1e-20)

            self.macro_precision = np.average([self.classwise_prec_recall_f1[k]['precision'] for k in self.classwise_correct_pred.keys()])
            self.macro_recall = np.average([self.classwise_prec_recall_f1[k]['recall'] for k in self.classwise_correct_pred.keys()])
            self.macro_f1 = np.average([self.classwise_prec_recall_f1[k]['f1'] for k in self.classwise_correct_pred.keys()])

            weights = [v.item() for v in self.classwise_target.values()]
            self.wgt_precision = np.average([self.classwise_prec_recall_f1[k]['precision'] for k in self.classwise_correct_pred.keys()], weights=weights)
            self.wgt_recall = np.average([self.classwise_prec_recall_f1[k]['recall'] for k in self.classwise_correct_pred.keys()], weights=weights)
            self.wgt_f1 = np.average([self.classwise_prec_recall_f1[k]['f1'] for k in self.classwise_correct_pred.keys()], weights=weights)

            return self.classwise_prec_recall_f1, self.micro_avg_f1, self.micro_prec, self.micro_recall, self.macro_f1, self.macro_precision, self.macro_recall, self.wgt_f1, self.wgt_precision, self.wgt_recall
        except Exception as e:
            raise e