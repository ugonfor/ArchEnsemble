import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMOModel(nn.Module):
    def __init__(self, backbone, ensemble_num: int = 3):
        super(MIMOModel, self).__init__()
        self.cnn_layer = backbone
        self.ensemble_num = ensemble_num
        self.last_head = nn.Linear(128, 10 * ensemble_num)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_shape_list = list(input_tensor.size())  # (ensemble_num,batch_size,1,28,28)
        ensemble_num, batch_size = input_shape_list[0], input_shape_list[1]
        assert ensemble_num == self.ensemble_num

        input_tensor = input_tensor.view([ensemble_num * batch_size] + input_shape_list[2:])
        output = self.cnn_layer(input_tensor)
        output = output.view(ensemble_num, batch_size, -1)
        output = self.last_head(output)
        output = output.view(ensemble_num, batch_size, ensemble_num, -1)
        output = torch.diagonal(output, offset=0, dim1=0, dim2=2).permute(2, 0, 1)
        output = F.log_softmax(output, dim=-1)
        return output
