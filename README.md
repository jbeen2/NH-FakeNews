# 설명자료 (Team 마이야르)

- 재현 가능한 코드들과 제출 파일들의 설명을 담고 있는 README 파일입니다.
- 데이터, 전처리 과정, 모델 및 추론 과정, 결과 파일이 포함되어 있습니다.
- 추론 과정은 `6.Code > NH_inference.ipynb`로 저장되어 있습니다.

## 1. 파일 구조

```python
마이야르
├── submission.csv
│
├── 1.Data     
│   ├───config.py
│   ├───news_train.csv
│   ├───news_test.csv
│   └── submission.csv
│
├── 2.Pos Tagger    
├── 3.Tokenizer
│
├── 4.Pre_trained embedding   # etri korbert 
│   ├───korbert-20210103T073731Z-001.zip
│   └── korbert
│       ├───vocab.korean_morp.list
│       ├───pytorch_model.bin
│       └───bert_config.json
│
├── 5.Model
│   ├───bert_jb   # huggingFace bert 
│   │   ├───pytorch_model.bin
│   │   ├───bert_config.json
│   │   ├───1230_bert_1.pt
│   │   └─── ...
│   │
│   ├───bert_tu   # ETRI korbert
│   │   ├───pytorch_model.bin
│   │   ├───bert_config.json
│   │   ├───test_results_labels.txt 
│   │   └─── ...
│   │
│   └── lgbm.pkl  # Machine Learning 
│
├── 6.Code 
│   └── NH_inference.ipynb
│
└───7.설명자료      
    └── README.md 

```

## 2. Inference Process 

세 가지의 모델을 통해 Inference가 진행되며, Ensemble을 통해 최종 결과값이 도출됩니다. 

### 1. HuggingFace bert 
* 


### 2. ETRI Korbert 
* 

### 3. LightGBM Classifier 
* 

### 4. Ensemble 
* 


