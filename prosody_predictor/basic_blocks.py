import torch.nn as nn
class ConvBasicBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel size should be odd but found {kernel_size=}"
        self.block = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = x.transpose(-1, -2)  # (B, T, C)->(B, C, T)
        y = self.block(x)
        y = y.transpose(-1, -2)  # (B, C, T)->(B, T, C)
        return y

class LNDropoutBasicBlock(nn.Module):
    def __init__(self, embed_dim, dropout_rate) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        y = self.block(x)
        return y

class Predictor(nn.Module):
    # Original settings for predictors can be referenced at appendix A of FastSpeech 2:
    # https://arxiv.org/abs/2006.04558
    def __init__(self, embed_dim=256, dropout_rate=0.1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBasicBlock(embed_dim, embed_dim, 9),
            LNDropoutBasicBlock(embed_dim, dropout_rate),
            ConvBasicBlock(embed_dim, embed_dim, 1),
            LNDropoutBasicBlock(embed_dim, dropout_rate),
            nn.Linear(embed_dim, 1),
        )
    
    def forward(self, x, mask):
        y = self.blocks(x)
        y = y.squeeze(-1)  # (B, T, 1)->(B, T)
        if mask is not None:
            y = y.masked_fill(mask, 0.0)
        return y