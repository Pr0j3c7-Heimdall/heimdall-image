import torch
import torch.nn as nn
import timm

class DualStreamConvNeXt(nn.Module):
    def __init__(self, num_classes=1):
        super(DualStreamConvNeXt, self).__init__()
        
        print("Initializing ConvNeXt V2 Tiny Backbones...")
        
        # 1. RGB Stream (이미지 담당)
        self.rgb_backbone = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
        
        # 2. Frequency Stream (LFS + FAD 담당)
        # 입력 채널이 2개(LFS+FAD)이므로 첫 레이어 수정 필요할 수 있으나
        # 편의상 pretrained 가중치 활용을 위해 Conv2d로 채널 뻥튀기 후 입력
        self.freq_backbone = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
        self.freq_entry = nn.Conv2d(2, 3, kernel_size=1) # 2채널 -> 3채널 변환
        
        # 3. Classifier
        # ConvNeXt Tiny의 출력 feature dim은 768
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, lfs, fad):
        # RGB Stream
        rgb_feat = self.rgb_backbone(rgb) # [Batch, 768]
        
        # Frequency Stream (Concat LFS & FAD)
        freq_input = torch.cat([lfs, fad], dim=1) # [Batch, 2, 256, 256]
        freq_input = self.freq_entry(freq_input)  # [Batch, 3, 256, 256]
        freq_feat = self.freq_backbone(freq_input) # [Batch, 768]
        
        # Feature Fusion
        combined = torch.cat([rgb_feat, freq_feat], dim=1) # [Batch, 1536]
        
        output = self.classifier(combined)
        return output