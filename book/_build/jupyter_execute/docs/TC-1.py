#!/usr/bin/env python
# coding: utf-8

# # HuggingFace Hub 을 활용한 Fine tuning Baseline
# 
# 
# - Task : KLUE-YNAT
# - 담당자: [김대웅](https://github.com/KimDaeUng) 님
# - 최종수정일: 21-08-25
# - 본 자료는 가짜연구소 3기 KLUE 로 모델 평가하기 크루 활동으로 작성됨
# 
# 
# 
# ## **이 노트북이 담고 있는 내용**
#     - HuggingFace Datasets을 활용하여 KLUE 데이터셋 쉽게 전처리하기
#     - HuggingFace Hub에서 사전학습된 언어 모델을 다운로드 받아 사용하고, 학습한 모델을 업로드하여 공유하기
#     - `Trainer` 객체를 사용하여 모델 학습 및 평가 & hyperparameter search하기
#     - [Weights & Biases](https://wandb.ai/)를 활용하여 실험 관리하기  
# 
# 
# ## **앞으로 추가되어야할 내용**
#     - Pretraining 직접 수행하기
#         - 학습 코퍼스 수집 및 전처리
#         - Pre-Tokenizer & Tokenizer
#         - Pretraining
# 
#     - Data Augmentation
#         - [Easy Data Augmentation(EDA)](https://arxiv.org/abs/1901.11196)
#         - Back Translation
#         - Summarization
#     
#     - Data Imbalance Problem
#         - `imbalanced-learn` 라이브러리를 활용한 over, under sampling
#         - Loss function: class weights 설정
#     
#     - 더 큰 배치사이즈로 학습하기
#         - Mixed Precision을 이용한 학습
#         - Single GPU에서 DeepSpeed 사용
# 
#     - 더 빠르게 학습하기 
#         - TPU를 이용한 학습
#         - Multi-GPU에서 DeepSpeed 사용
#         
# ---

# # 01 Init
# - KLUE-YNAT task를 다룰 때 필요한 기초적인 환경 설정 방법에 대해 설명합니다.
# ---

# ## Install packages
# 필요한 패키지를 설치합니다. 본 노트북에서는 3개의 패키지를 새로 설치해야 합니다.
# 
# transformers : hugging face(관련 페이지 링크) 에서 포팅된 데이터, 모델 등을 불러오기 위해 사용합니다.
# 
# datasets : hugging face 의 datasets 라이브러리(관련 페이지 링크) 중 load_dataset 매서드를 사용하면 쉽게 데이터를 다운로드 받을 수 있습니다.
# 
# wandb : 모델 학습 시 log 를 관리하기 위해 “Weight and Biases”(관련 페이지 [링크](https://wandb.ai/site)) 를 사용합니다.

# In[ ]:


get_ipython().system('pip install transformers datasets wandb')


# 본 노트북은 Transformers 4.9.2을 사용합니다.

# In[ ]:


import transformers
print(transformers.__version__)


# ## Login huggingface*
# huffingface hub를 이용하면 학습한 모델을 온라인 저장소에 올려 쉽게 공유할 수 있습니다. 이를 위해서는 먼저 회원가입이 필요합니다. [여기](https://huggingface.co/join)에서 회원가입을 진행할 수 있습니다. 
# 
# 그 다음, 아래 명령어를 실행시켜 Username과 Password를 입력합니다.

# In[ ]:


get_ipython().system('huggingface-cli login')


# huggingface hub는 대용량 파일을 저장하는 github repository의 개념입니다. 모델 parameter같은 대용량 파일 업로드를 위해 hf-lfs 설치와 git의 사용자 설정이 필요합니다. 이미 git 사용자 정보가 입력 되어있다면 hf-lfs만 설치하시면 됩니다.

# In[ ]:


get_ipython().system('pip install hf-lfs')
get_ipython().system('git config --global user.email eliza.dukim@gmail.com')
get_ipython().system('git config --global user.name KimDaeUng')


# In[ ]:


# argment setting
task = "ynat"
model_checkpoint = "klue/bert-base"
batch_size = 256


# # 02 Data Loading
# KLUE-YNAT task를 KLUE github에서 다운로드 받을수도 있지만 huggingface의 datasets 라이브러리를 사용하면 더욱 편리하게 이용할 수 있습니다.
# 
# ---

# ## Data Download
# huggingface의 [datasets](https://github.com/huggingface/datasets) 라이브러리를 사용해 데이터를 불러옵니다. `load_dataset`을 이용합니다.

# In[ ]:


from datasets import load_dataset
dataset = load_dataset('klue', 'ynat') # klue 의 task=nil 을 load
dataset # dataset 구조 확인


# ## Data view
# dataset 는 train 셋과 validation 으로 구성되어 있습니다.
# 
# 각각의 set의 구조를 데이터 샘플을 통해 살펴보겠습니다.

# In[ ]:


dataset['train'][0]


# KLUE TC (YNAT) task dataset 구조는 다음과 같습니다. 자세한 내용은 klue 공식 페이지 [링크](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-TC-(YNAT)-dataset-description)를 참조바랍니다.
#   1. 'date' : 뉴스 기사의 발행일
#   2. 'guid': index, 고유 식별자
#   3. 'label': 라벨
#   4. 'title': 뉴스 헤드라인
#   5. 'url': 뉴스의 url
# 
# 각 column 구성을 임의의 샘플을 추출하여 살펴보겠습니다.

# In[ ]:


import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

show_random_elements(dataset["train"])


# TC (YNAT) task 의 목적은 `title`이 어떤 토픽(Topic)에 속하는지 분류하는 것입니다. 따라서 input 값은 `title`이며 `label` 이 target 값으로 사용됩니다.
# 
# 다음은 데이터 전처리 과정을 살펴보겠습니다.
# 
# ----

# # 03 Data Processing
# KLUE-NLI task 중 Tokenizer를 사용하여 데이터를 인코딩한 후 전처리하는 과정을 설명합니다.
# 
# ---

# ## Tokenizer load
# 전처리를 위해 tokenizer로 데이터를 인코딩하는 과정이 필요합니다. transformers 라이브러리의 tokenizer 모듈을 이용해 모델의 입력 텍스트를 토크나이징하여 모델이 입력받는 포맷으로 변환할 수 있습니다.
# 
# `AutoTokenizer.from_pretrained`를 이용해 사용하는 모델과 관련된 tokenizer를 가져올 수 있습니다. 따라서 본 노트북에서는 KLUE의 bert base pretrianed model에서 사용된 tokenizer 을 활용하여 포팅된 모델을 사용하여 데이터를 인코딩 하겠습니다.

# In[ ]:


import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


# ## Data encoding
# 
# Tokenizer 는 문장 또는 문장 쌍을 입력하여 데이터에 대한 토크나이징을 수행할 수 있습니다.

# In[ ]:


dataset['train'][0]['title']


# In[ ]:


tokenizer(dataset['train'][0]['title'])


# In[ ]:


tokenizer(dataset['train'][0]['title']).tokens()


# YNAT task 에서는 `title`을 input으로 사용합니다. tokenizer의 입력값을 고려하여 전처리를 수행할 함수를 정의합니다. `tokenizer`에서 `truncation=True`를 설정하면 모델이 입력 최대 길이를 벗어난 경우 잘라냅니다. 

# In[ ]:


def preprocess_function(examples):
    return tokenizer(examples['title'], truncation=True)


# 이 함수를 전체 데이터셋의 문장에 적용시키기 위해서, `dataset` 객체의 `map` 메서드를 사용합니다. `batched=True` 옵션은 적용되는 함수가 배치형태로 처리가 가능한 함수인 경우에 체크하며, 멀티 쓰레딩을 사용해 텍스트 데이터를 배치 형태로 빠르게 처리할 수 있습니다.  
# 다음으로 hugging face 에 포팅된 Pretrained model 을 load 하여 fine tuning 하는 방법에 대해 다루겠습니다.
# 

# In[ ]:


encoded_dataset = dataset.map(preprocess_function, batched=True)


# # 04. Fine-tuning
# 
# KLUE-NLI task 중 Pretrained model 을 사용하여 fine-tuning 하는 방법에 대해 다루겠습니다.
# 
# 
# ---

# ## Model load
# 
# Pretrained model을 다운 받아 fine tuning 을 진행할 수 있습니다. YNAT task는 분류와 관련한 task 이므로 , `AutoModelForSequenceClassification` 클래스를 사용합니다.
# 
# 이때, label 개수에 대한 설정이 필요합니다.

# In[ ]:


# YNAT Task의 label 개수: 7
dataset['train'].features['label'].num_classes


# YNAT task 는 총 7개의 label 로 구성되어 있습니다.
# 
# 다음으로 모델을 불러오겠습니다. KLUE base 모델은 hugging face model hub (관련 사이트 [링크](https://huggingface.co/models)) 에 포팅되어 있으므로 model_checkpoint 경로를 정의하여 불러올 수 있습니다(`model_checkpoint = klue/bert-base` 로 사전 정의됨).
# 

# In[ ]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
num_labels = 7 # label 개수는 task 마다 달라질 수 있음
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                           num_labels=num_labels)


# > **\[참고\]**경고메시지의 의미
# - `BertForPreTraining`엔 있지만 `BertForSequenceClassification`엔 없는 레이어는 버리고
# - `BertForSequenceClassification` 엔 있지만 `BertForPreTraining`엔 없는 레이어는 랜덤 초기화.  
# - 따라서 `BertForSequenceClassification` 모델을 fine-tune 하지않으면 좋은 성능을 얻지 못 하니 fine-tune해서 써야 한다.[(reference)](https://github.com/huggingface/transformers/issues/5421)

# ## Parameter setting
# HuggingFace 에서는 `Trainer` 객체를 사용하여 학습을 진행합니다. 이때, `Trainer` 객체는 모델 학습을 위해 설정해야 하는 값이 들어있는 클래스인 `TrainingArgument`를 입력받아야 합니다.
# 
# 이번 단계에서는 모델 학습을 위한 trainer 객체를 정의하는 방법에 대해 다루겠습니다.
# 

# In[ ]:


import os

model_name = model_checkpoint.split("/")[-1]
output_dir = os.path.join("test-klue", "ynat") # task 별로 바꿔줄 것
logging_dir = os.path.join(output_dir, 'logs')
args = TrainingArguments(
    # checkpoint, 모델의 checkpoint 가 저장되는 위치
    output_dir=output_dir,
    overwrite_output_dir=True,

    # Model Save & Load
    save_strategy = "epoch", # 'steps'
    load_best_model_at_end=True,
    save_steps = 500,


    # Dataset, epoch 와 batch_size 선언
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    
    # Optimizer
    learning_rate=2e-5, # 5e-5
    weight_decay=0.01,  # 0
    # warmup_steps=200,

    # Resularization
    # max_grad_norm = 1.0,
    # label_smoothing_factor=0.1,


    # Evaluation 
    metric_for_best_model='eval_f1', # task 별 평가지표 변경
    evaluation_strategy = "epoch",

    # HuggingFace Hub Upload, 모델 포팅을 위한 인자
    push_to_hub=True,
    push_to_hub_model_id=f"{model_name}-finetuned-{task}",

    # Logging, log 기록을 살펴볼 위치, 본 노트북에서는 wandb 를 이용함
    logging_dir=logging_dir,
    report_to='wandb',

    # Randomness, 재현성을 위한 rs 설정
    seed=42,
)


# `TrainingArguments` 의 여러 인자 중 필수 인자는 `output_dir` 으로 모델의 checkpoint  가 저장되는 경로를 의미합니다.
# 
# 또한 task 별로 metric 지정이 필요합니다. YNAT task 는 Macro-F1을 평가지표로 사용합니다.
# 다음으로 `trainer` 객체를 정의하겠습니다. 우선 metric 설정이 필요합니다. `datasets` 라이브러리에서 제공하는 evaluation metric의 리스트를 확인하겠습니다.

# In[ ]:


# metric list 확인
from datasets import list_metrics, load_metric
metrics_list = list_metrics()
len(metrics_list)
print(', '.join(metric for metric in metrics_list))


# 이 중, YNAT 에서는 `f1` 를 사용합니다. 해당 평가지표를 고려하여 metric 계산을 위한 함수를 정의하겠습니다.

# In[ ]:


# YNAT의 metric은 F1 score를 사용합니다.
metric_macrof1 = load_metric('f1')

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return metric_macrof1.compute(predictions=predictions,
                                  references=labels, average='macro')


# 마지막으로 `Trainer` 객체를 정의하겠습니다.

# In[ ]:


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# 앞에서 `tokenizer`를 사용해 전처리를 했음에도 `Trainer`의 입력으로 다시 넣는 이유는 패딩을 적용해서 입력 샘플들을 동일한 길이로 만들기 위해 (데이터 로더의 마지막 과정에서) 사용하기 때문입니다. 모델에 따라 패딩에 대한 기본 설정이 다르기 때문에(왼쪽 패딩, 오른쪽 패딩, 또는 패딩 인덱스 번호 설정 등) `Trainer`는 이와 관련된 작업을 수행할 수 있는 `tokenizer`를 사용합니다.
# 
# 
# Fine-tuning 을 위한 준비가 완료되었습니다. 다음 단계에서는 fine-tuning 과 training log 를 관리하는 법에 대해 다루겠습니다.
# 
# ---

# # 05 Training
# 
# `Trainer` 객체를 이용하여 모델 학습을 하는 방법과 training log 를 관리하는 방법, 그리고 hyperparameter search 을 통해 모델 성능을 높이는 방법에 대해 다루겠습니다.
# 
# 
# ---

# ## Weights & Biases setting
# 
# huggingface는 모델 학습 로그를 기록할때 Tensorboard 또는 [Weights & Biases](https://wandb.ai/site)를 사용할 수 있습니다. 여기서는 Weights & Biases를 사용하겠습니다.
# 
# Weights & Biases를 사용하려면 먼저 회원가입이 되어있어야 합니다. 회원가입을 마친 후, https://wandb.ai/authorize 에서 얻은 key를 다음 셀에 입력하면 연동됩니다.
# 

# In[ ]:


import wandb
wandb.login()


# 실험 관리를 위해서 id값을 생성합니다. id는 각 실험에 부여되는 식별자입니다.

# In[ ]:


id = wandb.util.generate_id()
print(id)


# 생성된 id를 붙여 넣으면 `wandb`를 사용할 수 있습니다.
# > **\[참고\]** - project : 실험기록을 관리할 프로젝트 이름. 없을 시 입력받은 이름으로 생성, 여기선 예시로 klue로 설정
# - entity : weights & biases 사용자명 또는 팀 이름
# - id : 실험에 부여된 고유 아이디
# - name : 실험에 부여한 이름
# - resume : 실험을 재개할 떄, 실험에 부여한 고유 아이디를 입력

# In[ ]:


wandb.init(project='klue', # 실험기록을 관리한 프로젝트 이름
           entity='dukim', # 사용자명 또는 팀 이름
           id='3bso6955',  # 실험에 부여된 고유 아이디
           name='ynat',    # 실험에 부여한 이름               
          )


# ## Training
# 이제 `Trainer` 객체를 사용하여 학습을 진행할 수 있습니다.

# In[ ]:


trainer.train()


# 학습이 끝나면 `wandb` 도 종료합니다.

# In[ ]:


wandb.finish()


# 학습이 완료된 후, `evaluate` 메서드를 사용하여 `Trainer`가 best 모델로 불러온 모델의 성능을 확인해볼 수 있습니다.

# In[ ]:


trainer.evaluate()


# `push_to_hub()` 메서드를 사용하여 tokenizer를 비롯한 모델을 huggingface hub에 업로드할 수 있습니다.

# In[ ]:


trainer.push_to_hub()


# 업로드한 모델은 `huggingface hub 사용자 이름/사용자가 지정한 이름`으로 바로 다운로드하여 사용할 수 있습니다.

# In[ ]:


from transformers import AutoModelForSequenceClassification
# {HuggingFace Model Hub 사용자 아이디}/{push_to_hub_model_id에서 설정한 값}
model = AutoModelForSequenceClassification.from_pretrained('eliza-dukim/bert-base-finetuned-ynat', num_labels=num_labels)


# Submission을 위해 모델을 저장합니다.

# In[ ]:


trainer.save('/test-klue/ynat/model.h5')


# ## Hyperparameter search
# 
# 모델의 성능을 높이기 위해 Hyperparameter search를 수행할 수 있습니다. `Trainer` 는  optuna 또는 Ray Tune를 이용한 hyperparameter search를 지원합니다.
# 
# 우선 관련 라이브러리를 설치하겠습니다.

# In[ ]:


get_ipython().system(' pip install optuna')
get_ipython().system(' pip install ray[tune]')


# hyperparameter search 동안 `Trainer`는 학습을 여러 번 수행합니다. 따라서 모델이 매 학습마다 다시 초기화 될 수 있도록 모델이 함수에 의해 정의되도록 합니다.
# 
# 모델 초기화 함수를 정의하겠습니다.

# In[ ]:


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


# 모델 초기화 단계를 포함한 `Trainer`를 새롭게 정의하겠습니다. 이때, TrainingArguments 은 위에서 선언한 내용을 그대로 사용합니다.
# 
# 또한 hyperparameter search 과정에서 학습 시 시간이 오래 소요될 수 있습니다. 이 경우, `.shard(index=1, num_shards=10) ` 을 통해 일부 데이터 셋에 대한 hyperparameter 를 탐색할 수 있습니다.
# 
# num_shards 의 수에 따라 1/10, 1/5 데이터 만을 사용할 수 있습니다. 본 노트북에서는 1/5 데이터셋만 사용하여 hyperparameter 를 탐색하겠습니다. 
# 
# 물론, 탐색된 hyperparameter 는 전체 데이터에 대해 학습할 때 적용되어 최종 모델을 정상적으로 학습시킬 수 있습니다.

# In[ ]:


trainer_hps = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"].shard(index=1, num_shards=10), # 일부 데이터셋만 선택하여 진행 가능
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# `Trainer`정의가 완료되었다면, log 기록을 위해 wandb 를 다시 설정합니다. 위의 과정과 동일합니다.

# In[ ]:


wandb.init()
wandb.login()
id = wandb.util.generate_id()
print(id)


# wandb 에서 project 이름을 변경하여 wandb 를 초기화합니다. 아까와는 다르게 실험에 이름을 부여하지 않았는데, 여러번의 실험을 수행하면서 동일한 이름에 덮어씌워지지 않도록 하기 위함입니다.

# In[ ]:


wandb.init(project='klue', # 실험기록을 관리한 프로젝트 이름
           entity='dukim', # 사용자명 또는 팀 이름
           id='3bso6955',  # 실험에 부여된 고유 아이디
         # name='ynat',  # 실험에 이름을 부여하지 않으면 랜덤으로 생성함
          )


# 이제 `hyperparameter_search` 메서드를 사용해 hyperparameter search를 수행할 수 있습니다. 이 메서드는 `BestRun` 객체를 반환화는데, 최대화된 objective 값(평가지표값, 본 task에서는 Macro F1)과 이때 선택된 hyperparameter를 포함합니다.

# In[ ]:


best_run = trainer_hps.hyperparameter_search(n_trials=5, direction="maximize")


# 최종 best_run 에서 탐색된 hyperparameter 값은 다음과 같습니다.  여기서 선택된 hyperparameter를 이용해 전체 데이터셋에 대하여 위에서 소개된 절차로 다시 학습을 수행합니다.
# 

# In[ ]:


best_run


# # 06 Submission
# [작성중]
# 
# ---
# 

# In[ ]:


import sys
# if you have many scripts add this line before you import them
sys.path.append('/test-klue/ynat/') 


# In[ ]:


get_ipython().system('tar -czvf submission.tar.gz main.py model.actor.h5')


# # Appendix

# ## 0. `hyperparameter_search` 메서드 자세히 알아보기
# 
# - `hp_space` : hyperparameter search를 수행할 딕셔너리를 반환하는 함수를 입력받습니다. 값을 설정하지 않을 경우 optuna의 기본값을 사용합니다.
#  - optuna를 사용할 경우:  
# ```python
# def my_hp_space(trial):
#         return {
#             "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
#             "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
#             "seed": trial.suggest_int("seed", 1, 40),
#             "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
#         }
# ```
#  - ray를 사용할 경우:  
# ```python
# def my_hp_space_ray(trial):
#         from ray import tune
# 
#         return {
#             "learning_rate": tune.loguniform(1e-4, 1e-2),
#             "num_train_epochs": tune.choice(range(1, 6)),
#             "seed": tune.choice(range(1, 41)),
#             "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
#         }
# ```
# 
# 
# - `computive_objective` : 최대화하거나 최소화할 목적함수를 받습니다. 기본값으로 모델의 `evaluate` 메서드에 의해 반환되는 metric값(여기선 F1-score)를 사용합니다.
#     ```python
#     def my_objective(metrics):
#         return metrics["eval_f1"]
#     ```
# - `n_trials` : 테스트할 실험의 개수를 설정합니다(기본값 100).
# - `direction` : `computive_objective`값의 최적화의 방향을 정합니다. `'minimize'`(기본값) 또는 `'maximize'`.

# ## 1. optuna를 사용한 hyperparameter search code snippet

# ```python
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# 
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }
# 
# class MemorySaverCallback(TrainerCallback):
#     "A callback that deleted the folder in which checkpoints are saved, to save memory"
#     def __init__(self, run_name):
#         super(MemorySaverCallback, self).__init__()
#         self.run_name = run_name
# 
#     def on_train_begin(self, args, state, control, **kwargs):
#         print("Removing dirs...")
#         if os.path.isdir(f'./{self.run_name}'):
#             import shutil
#             shutil.rmtree(f'./{self.run_name}')
#         else:
#             print("\n\nDirectory does not exists")
# 
# training_args = TrainingArguments(
#     RUN_NAME, 
#     num_train_epochs=15,
#     per_device_train_batch_size=64,
#     per_device_eval_batch_size=64,
#     evaluation_strategy="epoch",
#     logging_strategy="steps",
#     logging_steps=1,
#     logging_first_step=False,
#     overwrite_output_dir=True,
#     save_strategy="no",
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_f1",
# )
# 
# trainer = Trainer(
#     model_init=partial(MyNet,2),
#     args=training_args, 
#     train_dataset=training_opos.select(range(2000)), 
#     eval_dataset=validating_opos,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=2), MemorySaverCallback(RUN_NAME)]
# )
# 
# def my_hp_space_optuna(trial):
#     return {
#         "learning_rate": trial.suggest_float("learning_rate", 2e-6, 2e-4, log=True),
#         "warmup_steps":  trial.suggest_float("warmup_steps", 0., 0.9, step=0.3),
#         "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-1)
#     }
# def my_objective(metrics):
#     return metrics["eval_f1"]
# 
# sa = trainer.hyperparameter_search(
#     direction="maximize", 
#     n_trials=1,
#     hp_space=my_hp_space_optuna, 
#     compute_objective=my_objective
# )
# ```

# # Reference
# - [Text Classification on GLUE](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb#scrollTo=71pt6N0eIrJo)을 KLUE의 주제 분류 데이터셋 YNAT에 맞게 수정 및 번역함.
# 
# - [HuggingFace Datasets Docs](https://huggingface.co/docs/datasets/index.html)
# - [HuggingFace Transformers Docs](https://huggingface.co/transformers/index.html)
# - [Using hyperparameter-search in Trainer](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/55)

# In[ ]:




