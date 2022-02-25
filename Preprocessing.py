#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install git+https://github.com/haven-jeon/PyKoSpacing.git


# In[22]:


# !pip install hanja --ignore-installed PyYAML


# In[28]:


from pykospacing import Spacing
import os
import pandas as pd
import string
import hanja
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


data_dir = os.path.join(os.getcwd(),'data')


# In[52]:


train_data = pd.read_csv(os.path.join(data_dir,'train.csv'))
test_data = pd.read_csv(os.path.join(data_dir,'test.csv'))


# In[56]:


train_data.dtypes


# In[6]:


spacing = Spacing()


# In[7]:


specialchars = "\'\"[]()“‘’”"
def title_processing(data):
    # 기사 제목에서 큰 따옴표, 작은 따옴표 [],() 제거
    for specialchar in specialchars:
        data = data.replace(specialchar,"")
    data = spacing(data)
    data = hanja.translate(data,'substitution')
    return data


# ## 기사 제목 processing

# * 기사 제목의 경우 맞춤법 검사는 하지 않음
# * 기사에 많이 쓰이는 따옴표, 한자를 수정함

# In[8]:


EDA_train_title = train_data['title'].apply(title_processing)


# In[9]:


EDA_train_title


# ## 댓글 processing

# * 댓글의 경우 따옴표등은 그대로
# * 중요한 부분은 띄어쓰기, 맞춤법, 신조어, 이모티콘 자모등 있다.

# In[ ]:


# !pip install soynlp


# In[10]:


from hanspell import spell_checker
from soynlp.normalizer import *


# In[11]:


def comment_processing(data):
    data = spell_checker.check(data).checked
    data = emoticon_normalize(data)
    data = spacing(a)
    return data


# In[29]:


EDA_train_comment = []
for i in tqdm(range(len(train_data['comment']))):
    data = train_data['comment'][i].replace('&',"")
    data = spell_checker.check(data).checked
    data = spacing(data)    
    data = emoticon_normalize(data)
    EDA_train_comment.append(data)


# In[45]:


EDA_train_comment = pd.Series(EDA_train_comment,name='comment',dtype='string')


# In[46]:


EDA_train = pd.concat([EDA_train_title,EDA_train_comment,train_data['bias'],train_data['hate']],axis=1)


# In[47]:


EDA_train.to_csv('./data/EDA_train.csv',index=False,encoding='utf-8')


# In[49]:


EDA_train.dtypes


# In[ ]:




