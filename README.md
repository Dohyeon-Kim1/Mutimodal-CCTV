# 📌 Multimodal-CCTV

## 🚶 프로젝트 소개
길을 걷다가 마주치는 무인매장들, 곳곳에 배치된 방범 CCTV 카메라들...  
하지만 기존의 영상 감시 시스템은 다음과 같은 한계를 가지고 있습니다.

- 감시 인력이 **직접 모니터링**해야 한다는 점  
- 사건이 발생한 **이후에만 대응이 가능**하다는 점  

이러한 문제를 해결하기 위해, **VLM(Visual Language Model) 기반의 멀티모달 CCTV 시스템**을 개발하였습니다.  
**자동으로 상황을 인식하고, 실시간 모니터링 정보를 갱신하는 AI CCTV**를 구현하여 감시 인력을 줄이고 더욱 신속한 대응이 가능하도록 하였습니다.

---

## ⚙ 구현 과정

###  1️⃣ 모델 선정

| 기능 | 모델 | 설명 |
|------|------|------|
| **객체 탐지 (Object Detection)** | YOLO | CNN 기반의 1-stage 탐지 모델로, 이미지 내 객체의 위치와 클래스를 실시간으로 예측 |
| **이미지 캡셔닝 (Image Captioning)** | BLIP | Transformer 기반 멀티모달 모델로, ViT 이미지 인코더와 BERT 언어 모델을 활용하여 이미지에 대한 설명(캡션) 생성 |
| **비디오 캡셔닝 (Video Captioning)** | VAST | Transformer 기반 모델로, 멀티모달 정보를 병합하여 비디오 내 행위를 이해하고 캡션 생성 |

---

###  2️⃣ 데이터셋

| 데이터셋 | 설명 |
|----------|------|
| **Carades Dataset** | 일상생활에서 수행하는 다양한 행동을 학습하기 위한 비디오 데이터셋 (9848개의 비디오 클립 및 스크립트 포함) → **BLIP & VAST 모델 학습에 활용** |
| **Kid_Image Dataset** | 다양한 연령대와 성별의 사람들이 가방, 책, 휴대폰 등과 상호작용하는 6,927개의 이미지 → **YOLO 모델 학습에 활용** |

---

## 🎯 주요 기능
### 🔍 **VLM 기술을 활용한 지능형 CCTV**
사람의 위험한 행동을 감지하여 경고를 주는 기능을 수행한다. 

object detection을 수행하여 사람이 감지된다면 관심있게 보고, 병렬적으로 image & video captioning을 통해 상황에 대한 캡션을 생성, 캡셔닝을 통해 사람의 행동을 추출하여 위험 행동을 효과적으로 감지.

---

## 🛠 개발 환경
- **언어 & 프레임워크**: `Python`, `PyTorch`, `OpenCV`
---

## 👨‍💻 팀원 소개 및 역할

| 이름 | 기수 | 역할 |
|------|------|------|
| **김도현** | 3기 | 객체 탐지 모델 훈련 |
| **심수민** | 2기 | 객체 탐지 모델 훈련 |
| **이예은** | 6기 | 비디오 캡셔닝 모델 훈련 |
| **호수빈** | 6기 | 이미지 캡셔닝 모델 훈련 |
| **박예은** | 6기 | 디자인 |

---

## 📌 실행 방법
```bash
# 1️⃣ 필수 라이브러리 설치
pip install -r requirements.txt

# 2️⃣ 모델 실행 (예시)
python main.py
