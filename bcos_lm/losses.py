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
        for idx in range(1, bsz):
            single_loss = 0
            mean = input[idx, :real_length[idx]].mean()
            for pos in range(1, real_length[idx]):
                if self.loss_type == 'L1':
                    single_loss += torch.abs(input[idx, pos] - input[idx, pos-1])
                elif self.loss_type == 'L2':
                    single_loss += (input[idx, pos] - input[idx, pos-1]) ** 2
                elif self.loss_type == 'auto_corr':
                    single_loss += (input[idx, pos] - mean) * (input[idx, pos-1] - mean)
                elif self.loss_type == 'neg_sum':
                    single_loss += -torch.sum(input[idx, pos] * input[idx, pos-1])
                else:
                    raise ValueError("Invalid loss type")
            loss += single_loss / real_length[idx]        
        return loss / bsz