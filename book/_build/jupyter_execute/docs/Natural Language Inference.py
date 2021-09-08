#!/usr/bin/env python
# coding: utf-8

# # Natural Language Inference

# - 본 자료는 가짜연구소 3기 KLUE 로 모델 평가하기 크루 활동으로 작성되었습니다.
# - 공식 KLUE 자료를 참고하여 진행되었습니다.
#     - [klue 홈페이지 (공식)](https://klue-benchmark.com/)
#     - [klue github (공식)](https://github.com/KLUE-benchmark/KLUE-baseline)
#     
# ---

# ## KLUE-NLI task
# - 목표 : 전제(premise)로 주어진 텍스트와 가설(hypothesis)로 주어진 텍스트 간의 관계를 추론
# - 데이터 구성 : 위키트리, 정책 뉴스 브리핑 자료, 위키뉴스, 위키피디아, 네이버 영화 리뷰, 에어비앤비 리뷰 등에서 조건에 부합하는 10000개의 전제를 추출, 약 30000개의 문장 쌍을 구성
# - label :
#     1. 전제와 가설간의 관계는 가설이 참인 경우(entailment)
#     2. 가설이 거짓인 경우(contradiction)
#     3. 가설이 참 일수도 있고 아닐 수도 있는 경우(neutral)로 라벨링
#     
# ---

# ## Notebook List
# 
# | Index | Task                       | Topic                                           | 담당자                                | 작성일   |
# | ----- | -------------------------- | ----------------------------------------------- | ------------------------------------- | -------- |
# | 03-1  | Natural language inference | Weight & Biase 을 활용한 Tokenizer 별 성능 비교 | [권지현](https://github.com/Jihyun22) | 21-08-18 |
# 
# 
# 
# 

# In[ ]:




