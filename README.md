# heimdall-image

Heimdall 프로젝트의 이미지 판별 모델 학습 결과물. Text-to-Image 생성 모델로 만들어진 이미지를 탐지(이진 분류)하고, 어떤 생성 모델로 만들어졌는지 추정(다중 분류)하는 세 가지 탐지 모델(DINOv3, F3Net, U-Net)로 구성되어 있습니다.

## 구성

```
binary/     # 이진 분류: Real vs AI-Generated
├── DINOv3 Code/
├── F3Net Code/
└── U-Net in sdv1.5 + K-NN or MLP code/
multiple/   # 다중 분류: 생성 모델 추정 (10개 생성 모델)
├── DINOv3 Code/
├── F3Net Code/
└── U-Net in sdv1.5 + K-NN or MLP code/
```

| 디렉토리 | 방식 | 비고 |
| --- | --- | --- |
| `DINOv3 Code/` | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)(Meta AI/FAIR)을 `transformers.AutoModel`로 불러와 동결된 특징 추출기로 사용, 자체 MLP 헤드(`mlp_model.py`)로 분류. 가중치는 실행 시 Hugging Face Hub에서 내려받으며 이 저장소에는 포함되지 않음 — 라이선스는 [NOTICE](NOTICE) 참고 |
| `F3Net Code/` | 자체 설계한 Dual-Stream 아키텍처. 주파수 영역(FAD/LFS 계열) 특징 + `timm`의 ConvNeXt V2(ImageNet 사전학습, Apache-2.0) 백본 조합. 방법론은 Qian et al., ECCV 2020 참고 |
| `U-Net in sdv1.5 + K-NN or MLP code/` | Stable Diffusion v1.5의 U-Net(`diffusers.StableDiffusionPipeline`)을 이미지 생성이 아닌 특징 추출기로만 사용, K-NN 또는 MLP 헤드로 분류. 가중치는 실행 시 Hugging Face Hub에서 내려받으며 이 저장소에는 포함되지 않음 — 라이선스는 [NOTICE](NOTICE) 참고 |

세 모델의 점수를 검증 정확도 기반 가중치로 Soft Voting 앙상블하여 최종 이진 판정을 산출합니다(DINOv3 34.95% · F3Net 46.28% · U-Net 18.77%, 최종 이진 분류 성능 99.8%).

각 모델 디렉토리는 `preprocess.py`/`extract_features.py`(특징 추출) → `train.py`(학습) → `test.py`/`evaluate.py`(평가) 순으로 구성되어 있고, 학습된 분류기 헤드(`models/*.pth`, `models/*.pkl`)와 평가 시 생성한 confusion matrix 이미지를 함께 담고 있습니다.

데이터셋 폴더 구조는 [dataset_directory.md](dataset_directory.md)에 정리되어 있습니다(원본 데이터셋 파일 자체는 용량 문제로 포함하지 않음).

## 데이터셋

- 총 1,100,000장 (학습 1,000,000 / 평가 100,000)
- 실제 사진 (학습 500,000 / 평가 50,000): BDD, COCO-2017, FGVC-Aircraft, Open-Image-V7, Openfake-Real, Oxford-IIIT-Pet, PASCAL-VOC-2012, Visual-Genome
- AI 생성 이미지: GPT-image-1, DALL·E 3, Stable Diffusion 3.5, Stable Diffusion XL, FLUX 1.1 Pro, BigGAN, GLIDE, Midjourney v6, Imagen 4, Nano-Banana — 모델별 학습 50,000 / 평가 5,000
- 실 서비스 환경(업로드 재압축 등)에 대한 강건성을 위해 원본(무압축) 40% / JPEG 재압축 60%(품질 95·85·75·65·50 구간별 차등 비율) 비율로 전처리

## 기술 스택

| 구분 | 내용 |
| --- | --- |
| 공통 | Python, PyTorch, numpy, scikit-learn, tqdm |
| DINOv3 | `transformers` |
| F3Net | `timm`(ConvNeXt V2 백본), OpenCV |
| U-Net | `diffusers`, `transformers`, `accelerate` |

## 라이선스

이 저장소의 코드는 [Apache License 2.0](LICENSE)으로 배포됩니다. DINOv3와 Stable Diffusion v1.5 U-Net은 코드가 아니라 실행 시 내려받는 사전학습 가중치이며 각각 별도 라이선스(Meta DINOv3 License, CreativeML OpenRAIL-M)를 따릅니다 — 자세한 내용은 [NOTICE](NOTICE)를 참고하세요.
