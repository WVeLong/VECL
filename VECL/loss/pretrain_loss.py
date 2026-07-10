import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class ThreeD_InfoNCE_Loss(nn.Module):
    def __init__(self):
        super(ThreeD_InfoNCE_Loss, self).__init__()

    def merge_and_deduplicate(self, input_list):
        new_list = []
        for list_sample in input_list:
            split_elements = set()
            for item in list_sample:
                elements = str(item).split(', ')
                for element in elements:
                    split_elements.add(element)
            new_list.append(list(split_elements))
        return new_list

    def creat_label(self, label_sample, label_list):
        target_p, target_0, target_n = [], [], []
        label_list = self.merge_and_deduplicate(label_list)
        mapping = {1: (1, 0, 0), 0: (0, 1, 0), -1: (0, 0, 1)}
        for list in label_list:
            row_list_p, row_list_0, row_list_n = [], [], []
            for sample in label_sample:
                if '25' not in sample and '26' not in sample:
                    label_temp = 1
                    elements = str(sample).split(', ')
                    for item in elements:
                        number = item[:-1]
                        sign = item[-1]
                        if sign == '-' and number + '+' not in list:
                            pass
                        elif sign == '+' and number + '+' in list:
                            pass
                        elif (sign == '+' and number + '-' in list) or (sign == '-' and number + '+' in list):
                            label_temp = -1
                        else:
                            if label_temp != -1:
                                label_temp = 0
                    row_list_p.append(mapping[label_temp][0])
                    row_list_0.append(mapping[label_temp][1])
                    row_list_n.append(mapping[label_temp][2])
                else:
                    row_list_p.append(0)
                    row_list_0.append(1)
                    row_list_n.append(0)
            target_p.append(row_list_p)
            target_0.append(row_list_0)
            target_n.append(row_list_n)
        for i in range(len(target_p)):
            target_p[i][i] = 1
            target_0[i][i] = 0
            target_n[i][i] = 0
        target = np.stack((target_p, target_0, target_n), axis=-1)
        return target

    def soft_cross_entropy_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        nan_mask = torch.isnan(target)
        valid_target = target[~nan_mask]
        valid_logprobs = logprobs[~nan_mask]
        cross_entropy_loss = 0
        if len(valid_target) != 0:
            cross_entropy_loss = -(valid_target * valid_logprobs).sum() / (valid_target.shape[0] / target.shape[0])
        return cross_entropy_loss

    def soft_infoNCE_loss(self, logits_per_img, soft_label):
        image_loss = self.soft_cross_entropy_loss(logits_per_img, soft_label / soft_label.sum(dim=1).unsqueeze(1))
        caption_loss = self.soft_cross_entropy_loss(logits_per_img.T, soft_label.T / soft_label.T.sum(dim=1).unsqueeze(1))
        return (image_loss + caption_loss) / 2

    def forward(self, input, label_sample, label_list):
        loss = 0
        target = torch.tensor(self.creat_label(label_sample, label_list)).float().to('cuda')
        slices = {'p': 0, '0': 1, 'n': 2}
        inputs = {key: input[:, :, idx] for key, idx in slices.items()}
        targets = {key: target[:, :, idx] for key, idx in slices.items()}
        loss += self.soft_infoNCE_loss(inputs['p'], targets['p']) # entailment
        loss += self.soft_infoNCE_loss(inputs['0'], targets['0']) # neutral
        loss += self.soft_infoNCE_loss(inputs['n'], targets['n']) # contradiction
        # return loss * 0.5

        # hybrid-loss

        # ① cross entropy loss, considering entailment (dim = 0)
        target_p = Variable(torch.LongTensor(range(len(input)))).to(input.device)
        loss += nn.CrossEntropyLoss()(inputs['p'], target_p) * 1.5
        loss += nn.CrossEntropyLoss()(inputs['p'].transpose(1, 0), target_p) * 1.5
        return loss * 0.25

        # # ② cross entropy loss, considering entailment and neutral (dim = 0, 1)
        # target_p = Variable(torch.LongTensor(range(len(input)))).to(input.device)
        # loss += nn.CrossEntropyLoss()(input[:, :, 0], target_p) * 0.75
        # loss += nn.CrossEntropyLoss()(input[:, :, 0].transpose(1, 0), target_p) * 0.75
        # tensor_0 = np.ones((len(input), len(input)))
        # np.fill_diagonal(tensor_0, 0)
        # tensor_0 = torch.from_numpy(tensor_0).float().to(input.device)
        # loss += self.soft_infoNCE_loss(inputs['0'], tensor_0) * 1.5
        # return loss * 0.25