import torch

class ConsecutiveLoss(torch.nn.Module):
    def __init__(self, loss_type='L1'):
        super(ConsecutiveLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, input):
        loss = 0
        bsz, seq_len = input.shape
        #mean = input.mean(dim=-1)
        real_length = torch.sum(input != 0, dim=-1)
        for idx in range(bsz):
            single_loss = 0
            seq = input[idx, :real_length[idx]]
            mean = seq.mean()
            if self.loss_type == 'L1':
                single_loss = torch.abs(seq[1:] - seq[:-1]).sum()
            elif self.loss_type == 'L2':
                single_loss = ((seq[1:] - seq[:-1]) ** 2).sum()
            elif self.loss_type == 'auto_corr':
                centered = input[idx, :real_length[idx]] - mean
                single_loss = -torch.sum(centered[1:] * centered[:-1])
            elif self.loss_type == 'neg_sum':
                single_loss += - torch.sum(seq[1:] * seq[:-1])
            else:
                raise ValueError("Invalid loss type")            
            loss += single_loss / real_length[idx]        
        return loss / bsz