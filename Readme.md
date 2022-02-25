# 기사 댓글 혐오, 편견 표현 분류학습

## 개요

기사제목과 댓글을 보고 편견, 혐오 여부 분류하는 모델링
### 파라미터 튜닝
    1. Tokenizer, Model 변경
    2. Pretraining Model을 한국어 model로 적용
    3. Hidden states 여러개를 반환해서 decoder에서 학습
