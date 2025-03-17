import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """将图像分割成patch并进行线性嵌入"""
    
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层实现patch embedding
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"输入图像大小 ({H}*{W}) 不等于预设大小 ({self.img_size}*{self.img_size})"
        
        # 投影并重塑
        # [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size] -> [B, embed_dim, n_patches] -> [B, n_patches, embed_dim]
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        
        return x


class Attention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        
        # 计算 QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, n_heads, N, head_dim]
        
        # 计算注意力矩阵
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, n_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权聚合
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """Feed Forward 网络"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 编码器块"""
    
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )
    
    def forward(self, x):
        # 自注意力层
        x = x + self.attn(self.norm1(x))
        # FFN层
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer 模型
    
    Args:
        img_size (int): 输入图像尺寸，默认为256
        patch_size (int): patch大小，默认为16
        in_channels (int): 输入图像通道数，默认为3 (RGB)
        num_classes (int): 分类类别数，默认为101
        embed_dim (int): 嵌入维度，默认为768
        depth (int): Transformer块数量，默认为12
        n_heads (int): 多头注意力中的头数，默认为12
        mlp_ratio (float): MLP中隐藏层维度与嵌入维度的比率，默认为4.0
        qkv_bias (bool): 是否在QKV投影中使用偏置，默认为True
        drop_rate (float): Dropout比率，默认为0.
        attn_drop_rate (float): 注意力Dropout比率，默认为0.
    """
    def __init__(self, img_size=256, patch_size=32, in_channels=3, num_classes=101,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches
        
        # 可学习的类别token和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer 编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, 
                n_heads=n_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # 层归一化和分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 初始化类别token
        nn.init.normal_(self.cls_token, std=0.02)
        # 初始化位置嵌入
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # 初始化所有线性层和层归一化
        self.apply(self._init_weights_layers)
    
    def _init_weights_layers(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        # Patch Embedding: [B, C, H, W] -> [B, n_patches, embed_dim]
        x = self.patch_embed(x)
        B, n_patches, _ = x.shape
        
        # 添加类别token: [B, n_patches+1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 层归一化
        x = self.norm(x)
        
        # 只使用类别token进行分类
        return x[:, 0]
    
    def forward(self, x):
        # 提取特征
        x = self.forward_features(x)
        # 分类
        x = self.head(x)
        return x
