# 기사 댓글 혐오, 편견 표현 분류학습

## 개요
기사제목과 댓글을 보고 편견, 혐오 여부 분류하는 모델링
### 모델 성능 향상 노력
    1. Tokenizer, Model 변경
    2. Pretraining Model을 한국어 model로 적용
    3. Hidden states 여러개를 반환해서 decoder에서 학습
    4. Text 전처리
    5. 추가 데이터 사용 및 optimizer, scheduler추가

## Pretraining, Model, Tokenizer, Hypter Parameter 별 Fine tunning 성능

### 통합 label로 학습했을 때

| Pre_training                          | Model                            | Tokenizer        | Optim | Scheduler  | batchsize | epoch | ACC   |
|---------------------------------------|----------------------------------|------------------|-------|------------|-----------|-------|-------|
| beomi/kcbert-base                     | BertForSequenceClassifier        | BertTokenizer    | adam  | get_linear_schedule_with_warmup       | 32        | 10    | 31.7  |
| skt/kobert-base-v1                    | BertForSequenceClassifier        | BertTokenizer    | adam  | get_linear_schedule_with_warmup       | 32        | 10    | 42.4  |
| skt/kobert-base-v1                    | BertForSequenceClassifier        | BertTokenizer    | adamW | get_cosine_schedule_with_warmup | 32        | 10    | 49.3  |
| beomi/beep-KcELECTRA-base                     | ElectraForSequenceClassification | ElectraTokenizer | adamW | get_cosine_schedule_with_warmup | 32        | 30    | 67.6  |
| kykim/electra-kor-base                | ElectraForSequenceClassification | ElectraTokenizer | adamW | get_cosine_schedule_with_warmup | 32        | 30    | 70.9  |
| monologg/koelectra-base-discriminator | ElectraForSequenceClassification | ElectraTokenizer | adamW | get_cosine_schedule_with_warmup | 32        | 50    | 64.09 |
| monologg/koelectra-base-discriminator | ElectraForSequenceClassification | ElectraTokenizer | adamW | get_cosine_schedule_with_warmup | 16        | 50    | 66.04 |

### Bias와 hate를 각각 모델링 후 통합
* bias성능은 충분하지만 hate성능이 충분하지 않아서 hate모델 위주로 최적화 시도

1. bias
| Pre_training                          | Model                            | Tokenizer        | Optim | Scheduler  | batchsize | epoch | ACC   |
|---------------------------------------|----------------------------------|------------------|-------|------------|-----------|-------|-------|
| monologg/koelectra-base-v3-bias       |ElectraForSequenceClassification  |ElectraTokenizer  |adamW  |get_cosine_schedule_with_warmup  |16         |50     |70.9|
|beomi/beep-KcELECTRA-base-bias|ElectraForSequenceClassification|ElectraTokenizer|adamW|OneCycleLR|16|50|71.7|




2. hate
| Pre_training                          | Model                            | Tokenizer        | Optim | Scheduler  | batchsize | epoch | ACC   |
|---------------------------------------|----------------------------------|------------------|-------|------------|-----------|-------|-------|
| monologg/koelectra-base-v3-hate-speech|ElectraForSequenceClassification  |ElectraTokenizer|adamW|OneCycleLR|16|50|64.09|
|beomi/beep-KcELECTRA-base-hate|ElectraForSequenceClassification|ElectraTokenizer|adamW|OneCycleLR|16|50|68.1|
|beomi/beep-KcELECTRA-base-hate_EDA|ElectraForSequenceClassification|ElectraTokenizer|adamW|OneCycleLR|16|50|67.01|

3. bias, hate통합
| Pre_training                          | Model                            | Tokenizer        | Optim | Scheduler  | batchsize | epoch | ACC   |
|---------------------------------------|----------------------------------|------------------|-------|------------|-----------|-------|-------|
| monologg bias, hate모델 결합          |ElectraForSequenceClassification  |ElectraTokenizer|adamW|OneCycleLR|16|50|70.07|
|beomi/beep-KcELECTRA-base-통합|ElectraForSequenceClassification|ElectraTokenizer|adamW|OneCycleLR|16|50|73.8|

## Text 전처리
   1. 띄어쓰기 처리 : [PyKospacing](https://github.com/haven-jeon/PyKoSpacing)
   2. 한자 변환 : [hanja](https://pypi.org/project/hanja/#description)
   3. 맞춤법 검사 : [hanspell](https://github.com/ssut/py-hanspell)
   4. 한글 이모티콘 normalization : [soynlp](https://github.com/lovit/soynlp)
