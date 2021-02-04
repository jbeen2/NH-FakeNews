# NH투자증권 빅데이터 경진대회 
### [League1] 알고리즘 개발 (NLP) : [AI야, 진짜 뉴스를 찾아줘!](https://dacon.io/competitions/official/235658/overview/) 
✔️ 분석 과정에 대해 자세하게 알고 싶으시다면, [**보고서**](https://github.com/jbeen2/NH-FakeNews/blob/main/League1_%EB%A7%88%EC%9D%B4%EC%95%BC%EB%A5%B4.pdf)를 참고해주세요! 



<table>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/43749571/106907658-d93b2800-6741-11eb-96a2-beb5303f9f3f.jpg" width="700px;" alt=""/></a></td>
  </tr>
</table>


- 재현 가능한 코드들과 제출 파일들의 설명을 담고 있는 README 파일입니다.
- 데이터, 전처리 과정, 모델 및 추론 과정, 결과 파일이 포함되어 있습니다.
- 추론 과정은 `6.Code > NH_inference.ipynb`로 저장되어 있으며, 해당 파일을 차례대로 실행하시면 됩니다.
- 모든 경로 정보는 `6.Code > config.py`에 있습니다.

<br>

## 1. 제출 파일 구조

```python
마이야르
├── submission.csv
│
├── 1.Data     
│   ├───news_train.csv
│   ├───news_test.csv
│   └── submission.csv
│
├── 2.Pos Tagger    
├── 3.Tokenizer
│
├── 4.Pre_trained embedding   # ETRI KorBERT  
│   ├───korbert-20210103T073731Z-001.zip
│   └── korbert
│       ├───vocab.korean_morp.list
│       ├───pytorch_model.bin
│       └───bert_config.json
│
├── 5.Model
│   ├───bert_jb   # Multilingual BERT 
│   │   ├───pytorch_model.bin
│   │   ├───bert_config.json
│   │   ├───1230_bert_1.pt
│   │   └─── ...
│   │
│   ├───bert_tu   # ETRI KorBERT
│   │   ├───pytorch_model.bin
│   │   ├───bert_config.json
│   │   ├───test_results_labels.txt 
│   │   └─── ...
│   │
│   └── lgbm.pkl  # Machine Learning 
│
├── 6.Code 
│   ├───config.py
│   ├───module.py
│   ├───preprocess.py
│   └── NH_inference.ipynb
│
└───7.설명자료      
    └── README.md 

```

<br>


## 2. Inference Process 

세 가지의 모델을 통해 Inference가 진행되며, Ensemble을 통해 최종 결과값이 도출됩니다. 

### 0. Preprocessing 
* ```news_train.csv```와 ```news_test.csv```에서 겹치는 중복 데이터셋의 경우, 진짜 뉴스와 광고성 뉴스라고 판단되어 미리 값을 고정하였습니다. 
* Multilingual BERT는 ```BertTokenizer```, ETRI KorBERT와 Machine Learning은 ```Mecab```을 사용해 Tokenizing 하였습니다. 

### 1. Multilingual BERT 
* 한자가 많은 데이터셋의 특성을 반영하여, ```bert-base-multilingual-cased``` 모델의 가중치를 이용해 Classification을 진행하였습니다. 
* 임의로 지정한 validation set에서의 accuracy는 0.99596, Dacon Public Score는 0.99064 입니다. 

### 2. ETRI KorBERT 
* 한국어 데이터셋의 특성을 반영할 수 있고, 30349개의 큰 단어집합을 가지고 있는 ```ETRI korBERT``` 모델의 가중치를 이용해 Classification을 진행하였습니다. 
* 임의로 지정한 validation set에서의 accuracy는 0.99479, Dacon Public Score는 0.98206 입니다. 

### 3. Machine Learning  
* 진짜 뉴스와 가짜 뉴스를 구분짓는 특성을 반영하는 Feature를 만들어, ```LightGBM Classifier```를 이용해 분류하였습니다. 
* 특정 문자의 포함 여부, 가짜 뉴스에 특히 많은 BAD Tokens의 개수, 기사 개수 및 순서, 해당 날짜의 진짜뉴스 및 가짜뉴스 비율 통계량 등의 Feature를 통해 Classification을 진행하였습니다. 
* 임의로 지정한 validation set에서의 accuracy는 0.9867 입니다. 

### 4. Ensemble 
* 두 개의 Bert 모델의 결과가 같으면 Bert의 결과를 따르고, 두 개의 결과가 다르면 Machine Learning의 결과를 따르는 방향으로 ```voting```을 진행하였습니다. 
* Dacon Public Score는 0.99126(21위), **Private Score**는 **0.98869(15위)** 입니다. 



<br>


## 3. Members : Team 마이야르 
고기가 그을리면서 특유의 감칠맛과 독특하고 그윽한 향을 갖추게 되는 것이 마이야르 반응이라고 합니다.. ㅎㅎ <br>
(항정살로 뭉쳐진 우리 팀 ..🐷)  

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable --> 

<table>
  <tr>
    <td align="center"><a href="https://github.com/taeukkkim"><img src="https://user-images.githubusercontent.com/43749571/106909675-db9e8180-6743-11eb-8eef-98972e3af2df.jpeg" width="150" height="150"><br /><sub><b>Taeuk Kim</b></sub></td>
    <td align="center"><a href="https://github.com/yoonjong12"><img src="https://user-images.githubusercontent.com/43749571/106909670-da6d5480-6743-11eb-982d-2536a4b538ac.png" width="150" height="150"><br /><sub><b>Yoonjong Shin</b></sub></td>
    <td align="center"><a href="https://github.com/jbeen2"><img src="https://user-images.githubusercontent.com/43749571/106911368-6764dd80-6745-11eb-83d3-bb105d794ab6.jpg" width="150" height="150"><br /><sub><b>Jaebeen Lee</b></sub></td>
  </tr>
</table>

<br>
