"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm
"""
import torch.nn
from torch.nn import Parameter

# import extension._bcnn as bcnn

__all__ = ['iterative_normalization', 'IterNorm']

class IterNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=-1, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True, *args, **kwargs):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim

        if num_channels == -1:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))

        if not self.affine:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, X: torch.Tensor):
        eps = 1e-5
        momentum = self.momentum

        nc = self.num_channels
        T = self.T
        g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(g, nc, -1)
        _, d, m = x.size()
        saved = []
        if self.training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            # calculate covariance matrix
            P = [None] * (T + 1)
            P[0] = torch.eye(d).to(X).expand(g, d, d)
            Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum(1, keepdim=True).sum(2, keepdim=True).reciprocal_()
            Sigma_N = Sigma * rTr
            for k in range(T):
                P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, P[k].bmm(P[k]).bmm(P[k]), Sigma_N)
            wm = P[T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
            self.running_mean += momentum * ( mean.detach() - self.running_mean)
            self.running_wm += momentum * ( wm.detach() - self.running_wm)

        else:
            xc = x - self.running_mean
            wm = self.running_wm
        xn = wm.matmul(xc)
        X_hat = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        
        # affine
        if self.affine:
            X_hat = X_hat * self.weight
            X_hat = X_hat + self.bias
        return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, dim={dim}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    ItN = IterNorm(64, num_groups=8, T=10, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    #x = torch.randn(32, 64, 14, 14)
    x = torch.randn(128, 64)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))
