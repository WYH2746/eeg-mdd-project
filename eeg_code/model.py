import torch, torch.nn as nn

class EEGNetAttn(nn.Module):
    """
    EEGNet-v2 + Multi-Head Self-Attention on flattened feature map.
    Input  : (B, C=channels, T=samples)
    Output : logits (B, 2)
    """
    def __init__(self, chans=128, samples=512, n_heads=4):
        super().__init__()
        self.block1 = nn.Sequential(     
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(        
            nn.Conv2d(16, 32, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        self.mhsa = nn.MultiheadAttention(32, n_heads, dropout=0.1, batch_first=True)
        self.fc   = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.block1(x)
        x = self.block2(x) 
        x = x.squeeze(2) 
        x = x.permute(0, 2, 1) 
        attn_out, _ = self.mhsa(x, x, x) 
        feat = attn_out.mean(dim=1)  
        return self.fc(feat)
