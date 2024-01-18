# 240118_TF

<!--  --> 

## 프로젝트 소개

### 프로젝트 제목

- 텐서플로우를 이용한 핵심 그래프신경망 모델(GCN, GraphSAGE, GAT, LightGCN) 구현

### 프로젝트 목적

- 텐서플로우를 이용한 신경망 구현 익히기

- 핵심 그래프신경망 모델인 GCN, GraphSAGE, GAT, LightGCN을 텐서플로우 코드 기반으로 익히기 

<!--  -->

## 주요 구현 내용

### 정보 정리

- CUDA (Compute Unified Device Architecture): GPU를 통한 계산 수행을 가능하게 해주는 라이브러리

- cuDNN (CUDA Deep Neural Network): 신경망을 위한 GPU 라이브러리 

- 연구실 서버(GRACE)의 경우 `GeForce 2080` 그래픽 카드 8장이 장착되어 있으며 해당 그래픽 카드의 Compute Capability는 7.5(Turning)있음 

- 따라서 연구실 서버의 그래픽 카드는 CUDA 10.0에서 12.3 버전의 라이브러리까지 호환이 가능함: 현재 `11.3` 버전의 CUDA를 설치하여 사용 중 

- `torch를` import하여 확인한 결과, GRACE는 `8.3.2` 버전의 cuDNN이 설치되어 있음

- 가상환경 내 pip를 이용하여 tensorflow-gpu 2.11 버전을 설치

<!--  -->

## 결과 분석

<!--  -->

## 참고 자료

- 핸즈온 머신러닝
