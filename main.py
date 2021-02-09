import torch.nn as nn
import torch.nn.functional as functional
import torch
import numpy as np

class deform_conv_v1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.p_conv = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size * kernel_size, padding=0,
                                kernel_size=kernel_size, stride=stride, bias=bias)
        
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.p_conv.bias, 0)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=kernel_size, padding=0, bias=bias)
        self.zero_padding = nn.ZeroPad2d(padding=padding)

    def forward(self, x):
        x = self.zero_padding(x)
        padded_h, padded_w = x.size()[2], x.size()[3]
        assert padded_h == padded_w
        offset = self.p_conv(x)

        n, _, h_out, w_out = list(offset.size())
        p = self.get_p(offset, h_out, w_out)
        p = p.permute([0, 2, 3, 1])
        p = torch.reshape(p, [n, h_out, w_out, 2, -1])
        p = torch.reshape(p.permute([0, 1, 2, 4, 3]), [n, h_out, w_out, self.kernel_size, -1, 2])
        
        grid = p.permute([0, 1, 3, 2, 4, 5])
        grid = torch.reshape(grid, [n, h_out * self.kernel_size, -1, 2])
        grid = torch.clamp(grid * 2 / (padded_h - 1) - 1, -1, 1)
        
        x = functional.grid_sample(x, grid, align_corners=True)
        x = self.conv(x)

        return x

    def get_p(self, offset, h_out, w_out):
        p_0 = self.get_p_0(h_out, w_out)
        p_0 = torch.reshape(p_0, [1, 2, self.kernel_size * self.kernel_size, h_out, w_out]).view(1, -1, h_out, w_out)
        p_n = self.get_p_n()

        # 1, 2k*k, 1, 1
        p_n = torch.reshape(p_n, [1, 2, -1, 1, 1]).view(1, -1, 1, 1)
        p = offset + p_0 + p_n
        return p

    def get_p_0(self, h_out, w_out):

        p_0_row, p_0_col = torch.meshgrid(
            torch.arange(start=self.padding, end=self.padding + h_out * self.stride, step=self.stride),
            torch.arange(start=self.padding, end=self.padding + w_out * self.stride, step=self.stride)
        )
        p_0_row = torch.flatten(p_0_row).view(1, 1, h_out, w_out).repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        p_0_col = torch.flatten(p_0_col).view(1, 1, h_out, w_out).repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        p_0 = torch.cat([p_0_col, p_0_row], 1)
        return p_0

    def get_p_n(self):
        p_n_row, p_n_col = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_col), torch.flatten(p_n_row)], 0)
        p_n = p_n.view(1, 2 * self.kernel_size * self.kernel_size, 1, 1)

        return p_n

if __name__ == '__main__':
    x = np.random.rand(8, 3, 32, 32).astype(np.float32)
    x = torch.from_numpy(x)

    model = deform_conv_v1(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)

    res = model(x)
    print (res)
