import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LocalAttention(nn.Module):
    """
    Local attention mechanism that restricts attention to nearby patches
    This is a modified version of multi-head attention that uses local attention masks
    """
    
    def __init__(self, embed_dim, num_heads, window_size=7, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def create_local_mask(self, seq_len, patch_size=14):
        """
        Create a local attention mask that restricts attention to nearby patches
        
        Args:
            seq_len: Sequence length (number of patches + 1 for CLS token)
            patch_size: Size of the patch grid (e.g., 14x14 for 224x224 image with 16x16 patches)
        
        Returns:
            mask: Boolean mask where True means attention is allowed
        """
        # For ViT, seq_len = num_patches + 1 (CLS token)
        # num_patches = (image_size // patch_size) ** 2
        num_patches = seq_len - 1
        grid_size = int(math.sqrt(num_patches))
        
        # Create mask
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # CLS token can attend to all patches and vice versa
        mask[0, :] = True  # CLS can attend to all
        mask[:, 0] = True  # All can attend to CLS
        
        # For patch tokens, create local attention windows
        for i in range(1, seq_len):  # Skip CLS token
            patch_idx = i - 1  # 0-indexed patch position
            row = patch_idx // grid_size
            col = patch_idx % grid_size
            
            # Define local window around current patch
            row_start = max(0, row - self.window_size // 2)
            row_end = min(grid_size, row + self.window_size // 2 + 1)
            col_start = max(0, col - self.window_size // 2)
            col_end = min(grid_size, col + self.window_size // 2 + 1)
            
            # Mark allowed attention positions
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    target_patch_idx = r * grid_size + c
                    target_seq_idx = target_patch_idx + 1  # +1 for CLS token
                    mask[i, target_seq_idx] = True
        
        return mask
    
    def forward(self, x, mask=None):
        """
        Forward pass with local attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        """
        B, N, C = x.shape
        
        # Create local mask if not provided
        if mask is None:
            mask = self.create_local_mask(N).to(x.device)
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # Apply local mask
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attn = attn.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class LocalTransformerBlock(nn.Module):
    """
    Transformer block with local attention
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, window_size=7, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LocalAttention(embed_dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LocalVisionTransformer(nn.Module):
    """
    Vision Transformer with local attention mechanism
    """
    
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, window_size=7, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.window_size = window_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks with local attention
        self.blocks = nn.ModuleList([
            LocalTransformerBlock(embed_dim, num_heads, mlp_ratio, window_size, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        x = self.head(x)
        
        return x


def create_local_attention_mask_visualization(seq_len=197, window_size=7, patch_size=14):
    """
    Create and visualize the local attention mask
    Useful for understanding the attention pattern
    """
    local_attn = LocalAttention(768, 12, window_size)
    mask = local_attn.create_local_mask(seq_len, patch_size)
    
    print(f"Attention mask shape: {mask.shape}")
    print(f"Window size: {window_size}")
    print(f"Percentage of allowed attention: {mask.float().mean().item():.3f}")
    
    return mask


def compare_attention_patterns():
    """
    Compare different window sizes for local attention
    """
    seq_len = 197  # 14*14 + 1 for CIFAR-10 with ViT-B/16
    window_sizes = [3, 5, 7, 9, 11]
    
    print("Local Attention Pattern Comparison:")
    print("=" * 50)
    
    for window_size in window_sizes:
        local_attn = LocalAttention(768, 12, window_size)
        mask = local_attn.create_local_mask(seq_len)
        attention_ratio = mask.float().mean().item()
        
        print(f"Window size {window_size:2d}: {attention_ratio:.3f} of attention allowed")
    
    # Full attention for comparison
    print(f"Full attention:     1.000 of attention allowed")


if __name__ == '__main__':
    # Test local attention
    print("Testing Local Attention Mechanism")
    print("=" * 40)
    
    # Create a small example
    batch_size = 2
    seq_len = 50  # 7*7 + 1
    embed_dim = 768
    
    # Test LocalAttention
    local_attn = LocalAttention(embed_dim, num_heads=12, window_size=3)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = local_attn(x)
    print(f"Output shape: {output.shape}")
    
    # Compare attention patterns
    compare_attention_patterns()
    
    # Test full LocalVisionTransformer
    print("\nTesting Local Vision Transformer")
    print("=" * 40)
    
    model = LocalVisionTransformer(
        img_size=224, patch_size=16, num_classes=10,
        embed_dim=768, depth=6, num_heads=12, window_size=7
    )
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Model input shape: {x.shape}")
    print(f"Model output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
