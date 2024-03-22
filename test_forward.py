import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.cuda.amp import autocast
import.pyplot as plt

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

def test_forward_time(module, input_shape, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    x = torch.randn(*input_shape).to(device)

    # 预热几次迭代，确保 CUDA 内核已经编译
    with autocast():
        for _ in range(10):_ = module(x)

    # 记录每个部分的耗时
    timings = {}

    for _ in range(num_iterations):
        with autocast():start_time = time.time()qkv = module.qkv(x)qkv_time = time.time() - start_timetimings["qkv"] = timings.get("qkv", 0) + qkv_timestart_time = time.time()qkv = qkv.view(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)q, k, v = qkv[0], qkv[1], qkv[2]qkv_reshape_time = time.time() - start_timetimings["qkv_reshape"] = timings.get("qkv_reshape", 0) + qkv_reshape_timestart_time = time.time()attn_weights = torch.einsum('bhnd,bhmd->bhnm', q, k) * module.scaleattn_weights = F.softmax(attn_weights, dim=-1)attn_weights_time = time.time() - start_timetimings["attn_weights"] = timings.get("attn_weights", 0) + attn_weights_timestart_time = time.time()gp = module.gp.weight.view(module.num_heads, module.gp_num, module.head_dim)group_weight = torch.einsum('bhnd,hmd->bhnm', q, gp)_, idx = torch.topk(group_weight, k=1, dim=-1)group_weight = torch.zeros_like(group_weight, memory_format=torch.channels_last).scatter_(dim=-1, index=idx, value=1)group_weight_time = time.time() - start_timetimings["group_weight"] = timings.get("group_weight", 0) + group_weight_timestart_time = time.time()q_mean = torch.einsum('bhng,bhnd->bhgd', group_weight, q)num_per_group = group_weight.sum(dim=2, keepdim=True).permute(0,1,3,2)q_mean /= num_per_group.clamp_(min=1e-8)q_mean_time = time.time() - start_timetimings["q_mean"] = timings.get("q_mean", 0) + q_mean_timestart_time = time.time()q_mean_weights = torch.einsum('bhgd,bhmd->bhgm', q_mean, k)_, idx = torch.topk(q_mean_weights, k=module.topk, dim=-1)q_mean_weights = torch.zeros_like(q_mean_weights, memory_format=torch.channels_last).scatter_(dim=-1, index=idx, value=1)q_mean_weights_time = time.time() - start_timetimings["q_mean_weights"] = timings.get("q_mean_weights", 0) + q_mean_weights_timestart_time = time.time()final = torch.einsum('bhng,bhgm->bhnm', group_weight, q_mean_weights)attn_weights = attn_weights * finalattn_weights /= attn_weights.sum(dim=-1, keepdim=True).clamp_(min=1e-8)attn_weights = module.attn_drop(attn_weights)final_time = time.time() - start_timetimings["final"] = timings.get("final", 0) + final_timestart_time = time.time()x = torch.matmul(attn_weights, v)x = x.view(B, module.num_heads, N, -1).transpose(1, 2).reshape(B, H, W, -1)matmul_time = time.time() - start_timetimings["matmul"] = timings.get("matmul", 0) + matmul_timestart_time = time.time()x = module.proj(x)x = module.proj_drop(x)proj_time = time.time() - start_timetimings["proj"] = timings.get("proj", 0) + proj_time

    # 计算平均耗时
    for key in timings:
        timings[key] /= num_iterations

    # 打印结果
    print("Forward pass timing results (mixed precision):")
    for key, value in timings.items():
        print(f"{key}: {value:.5f} seconds")

    # 绘制饼状图
    labels = list(timings.keys())
    sizes = list(timings.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title("Forward Pass Timing Results (Mixed Precision)")
    plt.show()

# 测试代码
B, H, W, C = 2, 32, 32, 512
input_shape = (B, H, W, C)
module = HardgroupAttentionV2(dim=C)
test_forward_time(module, input_shape)
