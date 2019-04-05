import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightDropout(nn.Module):
    """
    A module that replaces another layer's weights by 0 during training according to a probability p
    :param module: a nn.Module
    :param p: dropout probability of the weights: float
    :param name_w: list of name of weight matrices to take dropout of: list
    """

    def __init__(self, module, p=0.5, name_w=("weight_hh_l0")):
        super(WeightDropout, self).__init__()
        self.module, self.p, self.name_w = module, p, name_w
        for nw in name_w:
            # copy the weight parameters into this module
            w = getattr(self.module, nw)
            self.register_parameter(nw + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        # apply dropout to the raw weights and set them back into the module
        for nw in self.name_w:
            raw_w = getattr(self, nw + '_raw')
            self.module._parameters[nw] = F.dropout(raw_w, self.p, training=self.training)

    def forward(self, *args):
        self._setweights()
        self.module.flatten_parameters()
        return self.module.forward(*args)


if __name__ == '__main__':
    x = torch.randn((2, 1, 10)).to('cpu')
    lstm = nn.LSTM(10, 10)
    out = WeightDropout(lstm, p=0.9).to('cpu')

    test1 = [x.sum() for x in out(x)[0].data]

    print(out.weight_hh_l0_raw.sum())
    print(lstm.weight_hh_l0.sum())

    test2 = [x.sum() for x in out(x)[0].data]

    print(out.weight_hh_l0_raw.sum())
    print(lstm.weight_hh_l0.sum())

    print(test1)
    print(test2)

    assert (test1[0] == test2[0])
    assert (test1[1] != test2[1])
