import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Attention_qkv(nn.Module):
    """
    Optimized self-attention module using torch.nn.functional.scaled_dot_product_attention.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False, attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.q = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.attention_dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 使用torch.nn.functional.scaled_dot_product_attention进行优化
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop, is_causal=False, scale=self.scale)
        
        # 重塑和投影输出
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, H, W, self.attention_dim)
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x
    

class HardgroupAttention(nn.Module):

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Group parameter and additional functions
        self.gp_num = 48
        self.gp = nn.Linear(dim, self.gp_num, bias=False)
        self.topk = 96


    def forward(self, x):
        B, H, W, C = x.shape

        N = H * W
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(B, N, self.num_heads, -1).transpose(1, 2) for part in qkv]

        q, k, v = qkv
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        gp = self.gp.weight
        gp = gp.unsqueeze(0).view(self.num_heads, self.gp_num, self.head_dim)
        group_weight = torch.einsum('bhnd,hmd->bhnm', q, gp)
        _, idx = torch.topk(group_weight, k=1, dim=-1)
        group_weight = torch.zeros_like(group_weight)
        group_weight.scatter_(dim=-1, index=idx, value=1)
        q_mean = torch.einsum('bhng,bhnd->bhgd', group_weight, q)
        num_per_group = group_weight.sum(dim=2).unsqueeze(-1)
        q_mean = q_mean / num_per_group
        q_mean_weights = torch.einsum('bhgd,bhnd->bhng', q_mean, k)
        _, idx = torch.topk(q_mean_weights, k=self.topk, dim=-2)
        q_mean_weights = torch.zeros_like(q_mean_weights)
        q_mean_weights.scatter_(dim=-2, index=idx, value=1)            
        final = torch.einsum('bhng,bhmG->bhnm', group_weight, q_mean_weights)

        attn_weights = attn_weights * final
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)
        attn_weights = self.attn_drop(attn_weights)

        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class HardgroupAttentionV2(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False, attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Group parameter and functions
        self.gp_num = 48
        self.gp = nn.Linear(dim, self.gp_num, bias=False)
        self.topk = 96

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W

        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        gp = self.gp.weight.view(self.num_heads, self.gp_num, self.head_dim)
        group_weight = torch.einsum('bhnd,hmd->bhnm', q, gp)
        _, idx = torch.topk(group_weight, k=1, dim=-1)
        group_weight = torch.zeros_like(group_weight, memory_format=torch.channels_last).scatter_(dim=-1, index=idx, value=1)

        q_mean = torch.einsum('bhng,bhnd->bhgd', group_weight, q)
        num_per_group = group_weight.sum(dim=2, keepdim=True).permute(0,1,3,2)
        q_mean /= num_per_group.clamp_(min=1e-8)

        q_mean_weights = torch.einsum('bhgd,bhmd->bhgm', q_mean, k)
        _, idx = torch.topk(q_mean_weights, k=self.topk, dim=-1)
        q_mean_weights = torch.zeros_like(q_mean_weights, memory_format=torch.channels_last).scatter_(dim=-1, index=idx, value=1)

        final = torch.einsum('bhng,bhgm->bhnm', group_weight, q_mean_weights)

        attn_weights = attn_weights * final
        attn_weights /= attn_weights.sum(dim=-1, keepdim=True).clamp_(min=1e-8)
        attn_weights = self.attn_drop(attn_weights)


        x = x = torch.matmul(attn_weights, v)
        x = x.view(B, self.num_heads, N, -1).transpose(1, 2).reshape(B, H, W, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SoftgroupAttention(nn.Module):
    """
    Modified Softgroup Attention incorporating elements from MultiHeadAttention in Code B.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.g_space = nn.Linear(dim, dim)
        self.grouper = nn.Conv2d(dim, dim, kernel_size=3,stride=2,padding=1, groups=dim)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.group_gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        B, H, W, C = x.shape

        N = H * W
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = [part.reshape(B, N, self.num_heads, -1).transpose(1, 2) for part in qkv]

        q, k, v = qkv

        x_g = self.g_space(x)
        group_center = self.grouper(x_g.permute(0,3,1,2))
        x_g = x_g.reshape(B, N, self.num_heads, -1).transpose(1, 2)
        
        group_center = group_center.reshape(B, self.num_heads, self.head_dim, H//2, W//2).permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, N//4, self.head_dim)


        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                x_g,
                group_center
            )
        )
        
        
        
        group_weight = torch.matmul(sim, sim.transpose(-2, -1))
        gamma = torch.sigmoid(self.group_gamma)
        group_weight = group_weight * gamma + (1-gamma)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale  
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights * group_weight
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-8)
        attn_weights = self.attn_drop(attn_weights)

        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class AttentionDownsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x

if __name__ == '__main__':
    model = SoftgroupAttention(dim=64)

    img = torch.randn(2, 56, 56, 64)
    preds = model(img)
    print(preds.shape)