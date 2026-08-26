'''
train.py의 구조(레이어, 노드 수, 드롭아웃 등)를 관리하는 파일
'''
import torch.nn as nn

class DINOMlpClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=10):
        super(DINOMlpClassifier, self).__init__()
        
        # 기존 실험과 동일한 구조 (1024 -> 512 -> 256 -> 128 -> 64)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),        nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),        nn.BatchNorm1d(128),  nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),   nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)