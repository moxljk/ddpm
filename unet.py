import torch 
from torch import nn
import torch.nn.functional as F

Norm = nn.GroupNorm
Act = nn.SiLU

class TimeEncoder(nn.Module):
    def __init__(self, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim
    def forward(self, t):
        inv_freq = 1.0 / (
          10000 ** 
          (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = t[:, None]
        time_enc_s = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        time_enc_c = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        time_enc = torch.cat([time_enc_s, time_enc_c], dim=-1)
        return time_enc


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim, dropout) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Sequential(
            Norm(1, in_channels),
            Act(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.te = nn.Sequential(
            Norm(1, t_dim),
            nn.Linear(t_dim, out_channels)
        )        
        self.conv2 = nn.Sequential(
            Norm(1, out_channels),
            Act(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t):
        r = self.conv1(x)
        r += self.te(t)[:, :, None, None]
        r = self.conv2(r)
        return self.res_conv(x) + r


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = x
        # h = F.group_norm(x, C)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim, dropout) -> None:
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, t_dim, dropout)
        self.sa = AttnBlock(out_channels)
        self.res2 = ResBlock(out_channels, out_channels, t_dim, dropout)
    
    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.sa(x)
        x = self.res2(x, t)
        return x
    

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels_level:list[int]=None, attn_resolution=[16], t_dim=128, dropout=0.) -> None:
        super().__init__()
        self.attn_resolution = attn_resolution

        self.te = TimeEncoder(t_dim)
        self.down_sample = nn.AvgPool2d(2)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.down = nn.ModuleList()
        down_channels = [in_channels] + channels_level
        for i in range(1, len(channels_level), 1):
            self.down.append(ResBlock(down_channels[i-1], down_channels[i], t_dim=t_dim, dropout=dropout))
        
        self.mid = MiddleBlock(channels_level[-2], channels_level[-1], t_dim=t_dim, dropout=dropout)

        self.up = nn.ModuleList()
        for i in range(len(channels_level)-1, 0, -1):
            self.up.append(ResBlock(channels_level[i] + channels_level[i-1], channels_level[i-1], t_dim=t_dim, dropout=dropout))
        
        self.output = nn.Sequential(
            Norm(1, channels_level[0]),
            Act(),
            nn.Conv2d(channels_level[0], out_channels, kernel_size=3, padding=1)
        )

        self.attn_blocks = {}

    def forward(self, x, t):
        t = self.te(t)
        stack = []

        # down sampling
        for block in self.down:
            o = block(x, t)
            if o.shape[-1] in self.attn_resolution:
                try:
                    stack.append(self.attn_blocks[o.shape[-1]](o))
                except KeyError:
                    self.attn_blocks[o.shape[-1]] = AttnBlock(o.shape[1])
                    stack.append(self.attn_blocks[o.shape[-1]](o))
            else:
                stack.append(o)
            x = self.down_sample(o)

        x = self.mid(x, t)

        for block in self.up:
            x = self.up_sample(x)
            x = torch.concat((stack.pop(), x), 1)
            x = block(x, t)
        
        x = self.output(x)

        return x

