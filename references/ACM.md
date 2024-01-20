# [Revisiting Heterophily For Graph Neural Networks](https://arxiv.org/pdf/2210.07606.pdf)

- (Luan et al., 2022)

- NIPS 2022

## Abstract 

- 기존의 homophilhy metrics가 잘못되었음을 지적하고 `새로운 homophily metrics`를 제안함

- 위 과정을 통해 heterophily의 some harmful case가 `local diversification operation`에 의해 효과적으로 이용될 수 있음을 입증함 


## Introduction 


## Preliminaries


## Analysis of Heterophily 

### Motivation and Aggregation Homophily 

### Empirical Evaluation adn Comparison on Synthetic Graphs


## Adaptive Channel Mixing (ACM)

### Diversification Helps with Harmful Heterophily

### Filterbank and Adaptive Channel Mixing (ACM) Framework

- ACM-GCN 프레임워크 (No Variant, No Structure infomation)

    - 두개의 `GraphConvolution` 객체 레이어로 구성됨

    - 첫 번째 레이어는 (num feature x num hidden), 두 번째 레이어는 (num hidden X num class)로 구성됨

        - 요약의 편의를 위해 첫 번째 레이어의 구조를 기록 

    - 각 3개의 채널(low, high, mlp)에 대한 가중치 행렬과 가중치 벡터가 존재

        - 가중치 행렬(`weight_xxx`)은 (nfeat, nhid) shape으로 구성됨

        - 가중치 벡터(`att_vec_xxx`)는 (nhid, 1) shape으로 구성됨
    
        - 3-by-3 행렬인 `att_vec`은 채널별 어텐션 스코어를 조정함 
    
    - `forward` 함수에서는 각 채널과 feature matrix, weight matrix를 곱하여 임베딩 행렬(`output`)을 생성 

        - `attention3` 함수에 각 채널로부터 생성한 output 객체를 주입

        - 각 임베딩 행렬과 어텐션 벡터(att_vec)를 곱한 뒤, 각각을 concatenation하고 시그모이드 함수에 주입

        - 위 결과를 3-by-3의 어텐션 스코어 조정 행렬과 곱하여 logits 생성 

        - logits에 소프트맥스 함수를 취하고 최종적으로 (nnode, 1) shape의 노드별 어텐션 스코어 벡터를 반환 

        - 노드별 어텐션 스코어 벡터를 대각 원소로 하는 세 채널에 대한 대각행렬과 각 채널별 임베딩 행렬을 곱하여 모두 더한 뒤 3을 곱하며 forward 함수 반환 

            - 3을 곱하는 것이 의문

    - GraphConvolution 레이어가 반환한 결과에 ReLU 활성화 함수를 별도로 취함 

    - 두 번째 레이어는 클래스의 개수를 열의 개수로 하는 임베딩 행렬을 반환하며, 해당 행렬을 바로 소프트맥스 함수에 주입하여 노드 분류 정확도를 측정함 

## Related Work 


## Empirical Evaluation 