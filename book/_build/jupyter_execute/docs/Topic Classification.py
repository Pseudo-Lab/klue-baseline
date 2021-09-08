#!/usr/bin/env python
# coding: utf-8

# # YNAT(Topic Classification)

# - 본 자료는 가짜연구소 3기 KLUE 로 모델 평가하기 크루 활동으로 작성되었습니다.
# - 공식 KLUE 자료를 참고하여 진행되었습니다.
#     - [klue 홈페이지 (공식)](https://klue-benchmark.com/)
#     - [klue github (공식)](https://github.com/KLUE-benchmark/KLUE-baseline)
#     
# ---

# ## YNAT task
# - 목표 : 주어진 뉴스 표제(Headline)가 어떤 토픽(TopicP) __정치/경제/사회/생활문화/세계/IT과학/스포츠__에 속하는지 분류
# - 데이터 구성 : 2016년 12월 ~ 2020년 12월 네이버 뉴스에 게제된 연합뉴스 기자의 표제 70,000개 중 '정치/경제/사회/생활문화/세계/IT과학/스포츠'에 속한 표제를 각 토픽별로 고르게 추출, 개인정보 포함 문장 또는 분류 불가나 편견 및 혐오표현 문장을 제외한 63,892개의 문장으로 구성
# - label: 정치/경제/사회/생활문화/세계/IT과학/스포츠
# - train/val/test : 45,678 / 9,107 / 9,107 
#     
# ---

# ## Notebook List
# 
# | Index | Task                       | Topic                                           | 담당자                                | 작성일   |
# | ----- | -------------------------- | ----------------------------------------------- | ------------------------------------- | -------- |
# | 01-1  | YNAT(Topic Classification) | HuggingFace Hub 을 활용한 Fine tuning Baseline | [김대웅](https://github.com/KimDaeUng) | 21-08-25 |
# 
# 
# 
# 

# In[ ]:




