
# - 자연어 multi label classification 과제
# 
# 참고 논문 : 
# - [BERT: Pre-training of Deep Bidirectional Transformers for
# Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
# - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

# # 1. 환경 설정 및 라이브러리 불러오기

# In[1]:


# !pip install -r requirements.txt


# In[73]:


import pandas as pd
import os
import json
import numpy as np
import shutil

from sklearn.metrics import f1_score
from datetime import datetime, timezone, timedelta
import random
from tqdm import tqdm


from attrdict import AttrDict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils import *
from torch.optim import Adam, AdamW

from transformers import logging, get_linear_schedule_with_warmup


from transformers import ( 
    BertConfig,
    ElectraConfig
)

### v2 에서 라이브러리 추가됨
# 실험에 사용하실 모델 라이브러리를 추가하시는 걸 잊지 마세요!

from transformers import (
    BertTokenizer,  
    AutoTokenizer,
    ElectraTokenizer,
    AlbertTokenizer
)

from transformers import (
    BertModel,
    AutoModel, 
    ElectraForSequenceClassification,
    BertForSequenceClassification,
    AlbertForSequenceClassification
)
from kobert_tokenizer import KoBERTTokenizer


# In[74]:


# 사용할 GPU 지정
print("number of GPUs: ", torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
print("Does GPU exist? : ", use_cuda)
DEVICE = torch.device("cuda" if use_cuda else "cpu")


# In[75]:


# True 일 때 코드를 실행하면 example 등을 보여줌
DEBUG = False


# In[76]:


# config 파일 불러오기
config_path = os.path.join('config_v2.json')

def set_config(config_path):
    if os.path.lexists(config_path):
        with open(config_path) as f:
            args = AttrDict(json.load(f))
            print("config file loaded.")
            print(args.pretrained_model)
            print(args.run)
    else:
        assert False, 'config json file cannot be found.. please check the path again.'
    
    return args


# 코드 중간중간에 끼워넣어 리셋 가능
args = set_config(config_path)

# 결과 저장 폴더 미리 생성
os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(args.config_dir, exist_ok=True)


# # 2. EDA 및 데이터 전처리

# In[77]:


# data 경로 설정  
train_path = os.path.join(args.data_dir,'train.csv')

print("train 데이터 경로가 올바른가요? : ", os.path.lexists(train_path))


# ### 2-1. Train 데이터 확인

# In[78]:


train_df = pd.read_csv(train_path, encoding = 'UTF-8-SIG')

train_df.head()


# In[79]:


### v2 에서 추가됨

# title 중 가장 긴 타이틀 길이
max_len_title = np.max(train_df['title'].str.len())
max_len_title


# In[80]:


# comment 중 가장 긴 타이틀 길이
max_len_comment=np.max(train_df['comment'].str.len())
max_len_comment


# In[81]:


# 길이가 128이 넘는 코멘트 확인
train_df['comment'][train_df['comment'].str.len()>128]


# In[82]:


len(train_df)


# In[83]:


print("bias classes: ", train_df.bias.unique())
print("hate classes: ", train_df.hate.unique())


# In[84]:


pd.crosstab(train_df.bias, train_df.hate, margins=True)


# ### 2-2. Test 데이터 확인

# In[85]:


test_path = os.path.join(args.data_dir,'test.csv')
print("test 데이터 경로가 올바른가요? : ", os.path.lexists(test_path))


# In[86]:


test_df = pd.read_csv(test_path)
test_df.head()


# In[87]:


len(test_df)


# In[88]:


np.max(test_df['title'].str.len())


# In[89]:


np.max(test_df['comment'].str.len())


# ### 2-3. 데이터 전처리 (Label Encoding)
# bias, hate 라벨들의 class를 정수로 변경하여 라벨 인코딩을 하기 위한 딕셔너리입니다.

# - bias, hate 컬럼을 합쳐서 하나의 라벨로 만들기 

# In[90]:


# 두 라벨의 가능한 모든 조합 만들기
combinations = np.array(np.meshgrid(train_df.bias.unique(), train_df.hate.unique())).T.reshape(-1,2)

if DEBUG==True:
    print(combinations)


# In[91]:


# bias, hate 컬럼을 합친 것
bias_hate = list(np.array([train_df['bias'].values, train_df['hate'].values]).T.reshape(-1,2))

if DEBUG==True:
    print(bias_hate[:5])


# In[92]:


labels = []
for i, arr in enumerate(bias_hate):
    for idx, elem in enumerate(combinations):
        if np.array_equal(elem, arr):
            labels.append(idx)

train_df['label'] = labels
train_df.head()


# ## 3. Dataset 로드

# ### 3-0. Pre-trained tokenizer 탐색

# In[93]:


# config.json 에서 지정 이름별로 가져올 라이브러리 지정

TOKENIZER_CLASSES = {
    "BertTokenizer": BertTokenizer,
    "AutoTokenizer": AutoTokenizer,
    "ElectraTokenizer": ElectraTokenizer,
    "AlbertTokenizer": AlbertTokenizer,
    "Kobert" : KoBERTTokenizer
}


# - Tokenizer 사용 예시

# In[94]:


TOKENIZER = TOKENIZER_CLASSES[args.tokenizer_class].from_pretrained(args.pretrained_model)
if DEBUG==True:
    print(TOKENIZER)


# In[95]:


if DEBUG == True:
    example = train_df['title'][0]
    comment_ex = train_df['comment'][0]
    print(TOKENIZER(example, comment_ex))


# In[96]:


if DEBUG==True:
    print(TOKENIZER.encode(example),"\n")
    
    # 토큰으로 나누기
    print(TOKENIZER.tokenize(example),"\n")
    
    # 토큰 id로 매핑하기
    print(TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(example)))


# ### 3-1. Dataset 만드는 함수 정의

# In[97]:


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len, mode = 'train'):

        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        
        if self.mode!='test':
            try: 
                self.labels = df['label'].tolist()
            except:
                assert False, 'CustomDataset Error : \'label\' column does not exist in the dataframe'
     
    def __len__(self):
        return len(self.data)
                

    def __getitem__(self, idx):
        """
        전체 데이터에서 특정 인덱스 (idx)에 해당하는 기사제목과 댓글 내용을 
        토크나이즈한 data('input_ids', 'attention_mask','token_type_ids')의 딕셔너리 형태로 불러옴
        """
        title = self.data.title.iloc[idx]
        comment = self.data.comment.iloc[idx]
        
        tokenized_text = self.tokenizer(title, comment,
                             padding= 'max_length',
                             max_length=self.max_len,
                             truncation=True,
                             return_token_type_ids=True,
                             return_attention_mask=True,
                             return_tensors = "pt")
        
        data = {'input_ids': tokenized_text['input_ids'].clone().detach().long(),
               'attention_mask': tokenized_text['attention_mask'].clone().detach().long(),
               'token_type_ids': tokenized_text['token_type_ids'].clone().detach().long(),
               }
        
        if self.mode != 'test':
            label = self.data.label.iloc[idx]
            return data, label
        else:
            return data
        

    
train_dataset = CustomDataset(train_df, TOKENIZER, args.max_seq_len, mode ='train')
print("train dataset loaded.")


# In[98]:


if DEBUG ==True :
    print("dataset sample : ")
    print(train_dataset[0])


# In[99]:


# encoded_plus = tokenizer.encode_plus(
#                     sentence,                      # Sentence to encode.
#                     add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                     max_length = 128,           # Pad & truncate all sentences.
#                     pad_to_max_length = True,
#                     return_attention_mask = True,   # Construct attention masks.
#                     return_tensors = 'pt',     # Return pytorch tensors.
#                )


# ### 3-2. Train, Validation set 나누기

# In[100]:


from sklearn.model_selection import train_test_split
                                                         
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=args.seed)

train_dataset = CustomDataset(train_data, TOKENIZER, args.max_seq_len, 'train')
val_dataset = CustomDataset(val_data, TOKENIZER, args.max_seq_len, 'validation')

print("Train dataset: ", len(train_dataset))
print("Validation dataset: ", len(val_dataset))


# ## 4. 분류 모델 학습을 위한 세팅

# ### 4-1. 아키텍쳐 설정

# 
# 
# 
# - [PretrainedConfig](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)
# -[KcELECTRA 사전학습 모델](https://github.com/Beomi/KcELECTRA)

# In[101]:


from transformers import logging
logging.set_verbosity_error()

# config.json 에 입력된 architecture 에 따라 베이스 모델 설정
BASE_MODELS = {
    'BertModel':BertModel,
    "BertForSequenceClassification": BertForSequenceClassification,
    "AutoModel": AutoModel,
    "ElectraForSequenceClassification": ElectraForSequenceClassification,
    "AlbertForSequenceClassification": AlbertForSequenceClassification,

}


myModel = BASE_MODELS[args.architecture].from_pretrained(args.pretrained_model, 
                                                         num_labels = args.num_classes, 
                                                         output_attentions = False, # Whether the model returns attentions weights.
                                                         output_hidden_states = True # Whether the model returns all hidden-states.
                                                        )
if DEBUG==True:
    # 모델 구조 확인
    print(myModel)


# ### 4-2. 모델 설정

# 
# - BertForSequenceClassifier (line 1232부터 참고) [source code](https://github.com/huggingface/transformers/blob/a39dfe4fb122c11be98a563fb8ca43b322e01036/src/transformers/modeling_bert.py#L1284-L1287)
# 
# - ElectraForSequenceClassifier [source code](https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_electra.html#ElectraForSequenceClassification)
# 
# - SequenceClassier output 형태 : tuple(torch.FloatTensor)
#     - loss (torch.FloatTensor of shape (1,), optional, returned when label is provided):
#     Classification (or regression if config.num_labels==1) loss.
# 
#    - logits (torch.FloatTensor of shape (batch_size, config.num_labels)):
#     Classification (or regression if config.num_labels==1) scores (before SoftMax).
# 
#     - hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True):
#     Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
# 
#     - Hidden-states of the model at the output of each layer plus the initial embedding outputs.
# 
#     - attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True):
# Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
# 
#     Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
#     
#     
# - v2 에서 수정 및 추가 된 부분 많음
# 
# 

# In[102]:


### v2 에서 일부 수정됨
class myClassifier(nn.Module):
    def __init__(self, model, hidden_size = 768, num_classes=args.num_classes, selected_layers=False, params=None):
        super(myClassifier, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1) 
        self.selected_layers = selected_layers
        
        # 사실 dr rate은 model config 에서 hidden_dropout_prob로 가져와야 하는데 bert에선 0.1이 쓰였음
        self.dropout = nn.Dropout(0.1)


    def forward(self, token_ids, attention_mask, segment_ids):      
        outputs = self.model(input_ids = token_ids, 
                             token_type_ids = segment_ids.long(), 
                             attention_mask = attention_mask.float().to(token_ids.device))
        
        # hidden state에서 마지막 4개 레이어를 뽑아 합쳐 새로운 pooled output 을 만드는 시도
        if self.selected_layers == True:
            hidden_states = outputs.hidden_states
            pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
            # print("concatenated output shape: ", pooled_output.shape)
            ## dim(batch_size, max_seq_len, hidden_dim) 에서 가운데를 0이라 지정함으로, [cls] 토큰의 임베딩을 가져온다. 
            ## (text classification 구조 참고)
            pooled_output = pooled_output[:, 0, :]
            # print(pooled_output)

            pooled_output = self.dropout(pooled_output)

            ## 3개의 레이어를 합치므로 classifier의 차원은 (hidden_dim, 6)이다
            classifier = nn.Linear(pooled_output.shape[1], args.num_classes).to(token_ids.device)
            logits = classifier(pooled_output)
        
        else:
            logits=outputs.logits
        
    
        # 각 클래스별 확률
        prob= self.softmax(logits)
        # print(prob)
        # logits2 = outputs.logits
        # print(self.softmax(logits2))


        return logits, prob
        
# 마지막 4 hidden layers concat 하는 방법을 쓰신다면 True로 변경        
model = myClassifier(myModel, selected_layers=False)

# if DEBUG ==True :
#     print(model)


# ### 4-3. 모델 구성 확인

# In[103]:


if DEBUG==True:
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# ## 5. 학습 진행

# ### 5-0. Early Stopper 함수 정의
# 
# - v2에서 코드 일부 삭제

# In[104]:


class LossEarlyStopper():
    """Early stopper

        patience (int): loss가 줄어들지 않아도 학습할 epoch 수
        patience_counter (int): loss 가 줄어들지 않을 때 마다 1씩 증가
        min_loss (float): 최소 loss
        stop (bool): True 일 때 학습 중단

    """

    def __init__(self, patience: int)-> None:
        """ 초기화

        Args:
            patience (int): loss가 줄어들지 않아도 학습할 epoch 수
            weight_path (str): weight 저장경로
            verbose (bool): 로그 출력 여부, True 일 때 로그 출력
        """
        self.patience = patience
        self.patience_counter = 0
        self.min_loss = np.Inf
        self.stop = False

    def check_early_stopping(self, loss: float)-> None:
        # 첫 에폭
        if self.min_loss == np.Inf:
            self.min_loss = loss
           
        # loss가 줄지 않는다면 -> patience_counter 1 증가
        elif loss > self.min_loss:
            self.patience_counter += 1
            msg = f"Early stopping counter {self.patience_counter}/{self.patience}"

            # patience 만큼 loss가 줄지 않았다면 학습을 중단합니다.
            if self.patience_counter == self.patience:
                self.stop = True
            print(msg)
        # loss가 줄어듬 -> min_loss 갱신, patience_counter 초기화
        elif loss <= self.min_loss:
            self.patience_counter = 0
            ### v2 에서 수정됨
            ### self.save_model = True -> 삭제 (사용하지 않음)
            msg = f"Validation loss decreased {self.min_loss} -> {loss}"
            self.min_loss = loss

            print(msg)


# ### 5-1. Epoch 별 학습 및 검증

# - [Transformers optimization documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
# - [스케줄러 documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules)
# - Adam optimizer의 epsilon 파라미터 eps = 1e-8 는 "계산 중 0으로 나눔을 방지 하기 위한 아주 작은 숫자 " 입니다. ([출처](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/))
# - 스케줄러 파라미터
#     - `warmup_ratio` : 
#       - 학습이 진행되면서 학습률을 그 상황에 맞게 가변적으로 적당하게 변경되게 하기 위해 Scheduler를 사용합니다.
#       - 처음 학습률(Learning rate)를 warm up하기 위한 비율을 설정하는 warmup_ratio을 설정합니다.
#   

# In[105]:


args = set_config(config_path)

logging.set_verbosity_warning()

# 재현을 위해 모든 곳의 시드 고정
seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def train(model, train_data, val_data, args, mode = 'train'):
    
    # args.run은 실험 이름 (어디까지나 팀원들간의 버전 관리 및 공유 편의를 위한 것으로, 자유롭게 수정 가능합니다.)
    print("RUN : ", args.run)
    shutil.copyfile("config_v2.json", os.path.join(args.config_dir, f"config_{args.run}.json"))

    early_stopper = LossEarlyStopper(patience=args.patience)
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.train_batch_size)

    
    if DEBUG == True:
        # 데이터로더가 성공적으로 로드 되었는지 확인
        for idx, data in enumerate(train_dataloader):
            if idx==0:
                print("batch size : ", len(data[0]['input_ids']))
                print("The first batch looks like ..\n", data[0])
    
    
    criterion = nn.CrossEntropyLoss()
    
    total_steps = len(train_dataloader) * args.train_epochs

    ### v2에서 수정됨 (Adam -> AdamW)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_proportion), 
                                                num_training_steps=total_steps)

    
    if use_cuda:
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        

    tr_loss = 0.0
    val_loss = 0.0
    best_score = 0.0
    ### v2에서 변경됨
    best_loss = np.inf
      

    for epoch_num in range(args.train_epochs):

            total_acc_train = 0
            total_loss_train = 0
            
            assert mode in ['train', 'val'], 'your mode should be either \'train\' or \'val\''
            
            if mode =='train':
                for train_input, train_label in tqdm(train_dataloader):
                    
                    
                    mask = train_input['attention_mask'].to(DEVICE)
                    input_id = train_input['input_ids'].squeeze(1).to(DEVICE)
                    segment_ids = train_input['token_type_ids'].squeeze(1).to(DEVICE)
                    train_label = train_label.long().to(DEVICE)  
                    
                    ### v2에 수정됨
                    optimizer.zero_grad()
 
                    output = model(input_id, mask, segment_ids)
                    batch_loss = criterion(output[0].view(-1,6), train_label.view(-1))
                    total_loss_train += batch_loss.item()

                    acc = (output[0].argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc
                    
                    ### v2에 수정됨
                    optimizer.zero_grad()
                    
                    batch_loss.backward()
                    optimizer.step()
                    
                    ### v2 에 수정됨
                    scheduler.step()
                    

            total_acc_val = 0
            total_loss_val = 0
            
            # validation을 위해 이걸 넣으면 이 evaluation 프로세스 중엔 dropout 레이어가 다르가 동작한다.
            model.eval()
            
            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    mask = val_input['attention_mask'].to(DEVICE)
                    input_id = val_input['input_ids'].squeeze(1).to(DEVICE)
                    segment_ids = val_input['token_type_ids'].squeeze(1).to(DEVICE)
                    val_label = val_label.long().to(DEVICE)

                    output = model(input_id, mask, segment_ids)
                    ### v2 에서 일부 수정 (output -> output[0]로 myClassifier 모델에 정의된대로 logits 가져옴)
                    batch_loss = criterion(output[0].view(-1,6), val_label.view(-1))
                    total_loss_val += batch_loss.item()
                    
                    ### v2 에서 일부 수정 (output -> output[0]로 myClassifier 모델에 정의된대로 logits 가져옴)
                    acc = (output[0].argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            
            train_loss = total_loss_train / len(train_data)
            train_accuracy = total_acc_train / len(train_data)
            val_loss = total_loss_val / len(val_data)
            val_accuracy = total_acc_val / len(val_data)
            
            # 한 Epoch 학습 후 학습/검증에 대해 loss와 평가지표 (여기서는 accuracy로 임의로 설정) 출력
            print(
                f'Epoch: {epoch_num + 1} \
                | Train Loss: {train_loss: .3f} \
                | Train Accuracy: {train_accuracy: .3f} \
                | Val Loss: {val_loss: .3f} \
                | Val Accuracy: {val_accuracy: .3f}')
          
            # early_stopping check
            early_stopper.check_early_stopping(loss=val_loss)

            if early_stopper.stop:
                print('Early stopped, Best score : ', best_score)
                break

            ### v2 에 수정됨
            ### loss와 accuracy가 꼭 correlate하진 않습니다.
            ### 
            ### 원본 (필요하다면 다시 해제 후 사용)
            # if val_accuracy > best_score : 
            if val_loss < best_loss :
            # 모델이 개선됨 -> 검증 점수와 베스트 loss, weight 갱신
                best_score = val_accuracy 
                
                ### v2에서 추가
                best_loss =val_loss
                # 학습된 모델을 저장할 디렉토리 및 모델 이름 지정
                SAVED_MODEL =  os.path.join(args.result_dir, f'best_{args.run}.pt')
            
                check_point = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(check_point, SAVED_MODEL)  
              
            # print("scheduler : ", scheduler.state_dict())


    print("train finished")


train(model, train_dataset, val_dataset, args, mode = 'train')


# ## 6. Test dataset으로 추론 (Prediction)
# 
# 
# - v2 에서 수정된 부분
#     - output -> output[0]

# In[106]:


from torch.utils.data import DataLoader

# 테스트 데이터셋 불러오기
test_data = CustomDataset(test_df, tokenizer = TOKENIZER, max_len= args.max_seq_len, mode='test')

def test(model, SAVED_MODEL, test_data, args, mode = 'test'):


    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size)


    if use_cuda:

        model = model.to(DEVICE)
        model.load_state_dict(torch.load(SAVED_MODEL)['model'])


    model.eval()

    pred = []

    with torch.no_grad():
        for test_input in test_dataloader:

            mask = test_input['attention_mask'].to(DEVICE)
            input_id = test_input['input_ids'].squeeze(1).to(DEVICE)
            segment_ids = test_input['token_type_ids'].squeeze(1).to(DEVICE)

            output = model(input_id, mask, segment_ids)

            output = output[0].argmax(dim=1).cpu().tolist()

            for label in output:
                pred.append(label)
                
    return pred

SAVED_MODEL =  os.path.join(args.result_dir, f'best_{args.run}.pt')

pred = test(model, SAVED_MODEL, test_data, args)


# In[107]:


print("prediction completed for ", len(pred), "comments")


# ### 

# In[108]:


# 0-5 사이의 라벨 값 별로 bias, hate로 디코딩 하기 위한 딕셔너리
bias_dict = {0: 'none', 1: 'none', 2: 'others', 3:'others', 4:'gender', 5:'gender'}
hate_dict = {0: 'none', 1: 'hate', 2: 'none', 3:'hate', 4:'none', 5:'hate'}

# 인코딩 값으로 나온 타겟 변수를 디코딩
pred_bias = ['' for i in range(len(pred))]
pred_hate = ['' for i in range(len(pred))]

for idx, label in enumerate(pred):
    pred_bias[idx]=(str(bias_dict[label]))
    pred_hate[idx]=(str(hate_dict[label]))
print('decode Completed!')


# In[109]:


submit = pd.read_csv(os.path.join(args.data_dir,'sample_submission.csv'))
submit


# In[110]:


submit['bias'] = pred_bias
submit['hate'] = pred_hate
submit


# In[111]:


submit.to_csv(os.path.join(args.result_dir, f"submission_{args.run}.csv"), index=False)


# In[ ]:





# In[ ]:




