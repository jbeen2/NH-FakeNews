import re
import pandas as pd

def clean_text(text):
    '''
    :param text: string
    '''
    text = re.sub('↑', '상승 ', text)
    text = re.sub('↓', '하락 ', text)
    text = re.sub('㈜', '주식회사 ', text)
    return text
    
def get_pos(x, tokenizer):
    '''
    :param x: string
    :param tokenizer: POS tagger (Mecab)
    '''
    tk = []
    token = tokenizer.pos(x)

    for t in token:
        if t[1] == 'NNBC':
          tk += [t[0] + "/" + "NNB" + "_"]
        elif  (t[1] == 'SSO') or (t[1] == 'SSC'):
          tk += [t[0] + "/" + "SS" + "_"]
        elif t[1] == 'SY':
          tk += [t[0] + "/" + "SW" + "_"]
        else: 
          tk += [t[0] + "/" + t[1] + "_"]
          
    return tk
  
def count_badtokens(tokens, bad_tokens) : 
    '''
    :param tokens: list of token
    :param bad_tokens: list of bad tokens
    '''
    cnt = 0 
    for token in tokens : 
        if token in bad_tokens.keys() : 
            cnt += 1 
    return cnt 
    
def jaccard_sim(A, B):
    '''
    :param A: string
    :param B: string
    
    :return: jaccard similarity between A and B
    '''
    A = set(A)
    B = set(B)
    intersect = A.intersection(B)
    denom = (len(A) + len(B) - len(intersect))
    if not denom:
        return 0
    
    return len(intersect) / denom
    
def construct_vocab(file_dir, max_size=200000, mincount=5):
    vocab2id = {'[CLS]': 2, '[SEP]': 3, '[PAD]': 0, '[UNK]': 1, '[STOP]': 4}
    word_pad = {'[CLS]': 2, '[SEP]': 3, '[PAD]': 0, '[UNK]': 1, '[STOP]': 4}
    
    cnt = len(vocab2id)
    with open(file_dir, 'r') as fp:
      for line in fp:
        arr = re.split('\t', line[:-1])
        if (arr[0] in [' ', 'n_iters=10000', 'max_length=16', '[MASK]','<S>','<T>']) :
          continue
        if arr[0] in word_pad:
          continue
        if int(arr[1]) >= mincount:
          vocab2id[arr[0]] = cnt
          cnt += 1
        if len(vocab2id) == max_size:
          break
  
    return vocab2id
