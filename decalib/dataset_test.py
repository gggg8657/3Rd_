class Config:
    K = 4
    image_size = 224
    mediapipePath ='/home/cine/Documents/DJ/DECA/data/mediapipe_landmark_embedding.npz'
    batch_size = 4

from torch.utils.data import DataLoader
from .datasets import build_datasets_detail as build_datasets
# 데이터 로더 설정
config = Config()
train_dataset = build_datasets.build_train(config)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size // config.K, shuffle=True, num_workers=4, pin_memory=True)

for batch in train_loader:
    print("Batch image shape:", batch['image'].shape)
    print("Batch mask shape:", batch['mask'].shape)
    print("Batch kpts shape:", batch['kpts'].shape)
    print("Batch dense_kpts shape:", batch['dense_kpts'].shape)
    break