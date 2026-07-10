import torch
import torch.nn as nn


class CLASSIFYLoss(nn.Module):
    def __init__(self):
        super(CLASSIFYLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, label):
        label = label.to(input.device)
        loss = self.bce_loss(input, label.float())
        return loss * 10


class CLASSIFY_PLUS_Loss(nn.Module):
    def __init__(self):
        super(CLASSIFY_PLUS_Loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def process_label(self, label_temp):
        mapping = {1: (1, 0, 0), 0: (0, 1, 0), -1: (0, 0, 1)}
        mapping_tensor = torch.tensor([mapping[0], mapping[1], mapping[-1]], device=label_temp.device, dtype=label_temp.dtype)
        expanded_tensor = label_temp.unsqueeze(-1).expand(-1, -1, 3)
        output_tensor = torch.zeros_like(expanded_tensor, device=label_temp.device)
        output_tensor[label_temp == 0] = mapping_tensor[0]
        output_tensor[label_temp == 1] = mapping_tensor[1]
        output_tensor[label_temp == -1] = mapping_tensor[2]
        return output_tensor

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

    def forward(self, input, label):
        target = self.process_label(label)
        loss = 0

        # 3D InfoNCE Loss
        slices = {'p': 0, '0': 1, 'n': 2}
        inputs = {key: input[:, :, idx] for key, idx in slices.items()}
        targets = {key: target[:, :, idx] for key, idx in slices.items()}
        loss += self.soft_infoNCE_loss(inputs['p'], targets['p'])
        loss += self.soft_infoNCE_loss(inputs['0'], targets['0'])
        loss += self.soft_infoNCE_loss(inputs['n'], targets['n'])

        return loss