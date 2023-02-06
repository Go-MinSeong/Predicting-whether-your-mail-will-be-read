#!/usr/bin/env python
# coding: utf-8

# ## **사용자 기반 네이버 메일 알림 시스템**

# In[1]:


# 시각화를 위한 함수 정의
def print_term_topics(term, dictionary, model):
    word_id = dictionary.token2id[term]   #단어의 아이디 구함
    print(model.get_term_topics(word_id))  
    
# 문서에 대한 토픽가중치를 반복하면서 전체 문서에 대해서 표시
def print_doc_topics(model, corpus): 
    
    for doc_num, doc in enumerate(corpus):
        topic_probs = model[doc]
        print("Doc num: {}".format(doc_num))

        for topic_id, prob in topic_probs:
            print("\t{}\t{}".format(topic_id, prob))
        
        if doc_num == 2:  # 시간 관계상 2번 문서까지만 출력, "0번문서, 1번문서, 2번문서"에 대해서만 해당문서의 토픽가중치를 표시                                     
            break

        print("\n")  

# ★ 모델링 후 각 토픽별로 중요한 단어들을 표시
def print_topic_words(model):
    
    for topic_id in range(model.num_topics):
        topic_word_probs = model.show_topic(topic_id, NUM_TOPIC_WORDS)
        print("Topic ID: {}".format(topic_id))

        for topic_word, prob in topic_word_probs:
            print("\t{}\t{}".format(topic_word, prob))
        print("\n")


# In[2]:


# 사용 모듈
import os
import warnings
import time
from tqdm import tqdm
import gc
import shap

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import matplotlib as mpl
from matplotlib import font_manager
from PIL import Image
import matplotlib.font_manager as fm 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rc('font', family='Malgun Gothic')  

import pandas as pd
import numpy as np

from selenium import webdriver
from bs4 import BeautifulSoup
import pyperclip

import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from selenium.webdriver.common.keys import Keys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt ; okt=Okt()
from collections import Counter
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *
from imblearn.ensemble import BalancedRandomForestClassifier


from gensim import corpora #gensim 에서 제공하는 패키지
from gensim import models
from gensim.models import CoherenceModel
from collections import Counter #카운터 사용
import pyLDAvis.gensim_models # LDA시각화


# #### 네이버 메일 자동 로그인
# - 자신의 아이디와 비밀번호를 해당 칸에 입력해야 로그인 가능

# In[4]:


# 드라이버 설정
driver = webdriver.Chrome('C:\chromedriver.exe')
time.sleep(0.5)
driver.implicitly_wait(3)

# 페이지 설정
driver.get('https://mail.naver.com')

# 네이버 자동 로그인
driver.find_element_by_name('id').click()

# 자동 로그인 우회 방식
pyperclip.copy('minsung0321') # 아이디
driver.find_element_by_name('id').send_keys(Keys.CONTROL, 'v')
driver.find_element_by_name('pw').click() # 패스워드
pyperclip.copy('**********')
driver.find_element_by_name('pw').send_keys(Keys.CONTROL, 'v')

# 로그인 클릭
driver.find_element_by_id('log.login').click()

html = driver.page_source 
soup = BeautifulSoup(html, 'html.parser')

# 팝업창 닫기
#driver.find_element_by_xpath("//*[@id='smartFolderLayer']/div/div/button").send_keys(Keys.ENTER)


# #### 사용자의 메일 데이터를 수집
# - 메일함 데이터 수집 (동적 크롤링) 5000개당 약 한시간 소요

# In[4]:


# 페이지 넘기기
k=4
data=[[] for _ in range(4)]
contents_list=[]
for _ in tqdm(range(500)):  # 100 페이지 당 2시간 소요 
    # 페이지가 넘어갈 때마다 크롤링 재설정
    sendlist = driver.find_elements_by_css_selector('div.name')
    time.sleep(1)
    maillist = driver.find_elements_by_css_selector('strong.mail_title')
    time.sleep(1)
    read_jugements = driver.find_elements_by_css_selector('li.read > a')
    time.sleep(1)
    dates = driver.find_elements_by_css_selector('li.iDate')
    time.sleep(1)
    #driver.find_elements_by_css_selector('div.subject').click()
    #time.sleep(1)
    # 데이터 수집     발신인 , 제목
    

    for mail in zip(sendlist, maillist, dates, read_jugements):
        for j in range(3):
            data[j].append(mail[j].text)
        data[3].append(mail[3].get_attribute("title"))
    time.sleep(1)
    
    # 셀레니움을 이용해 페이지를 왔다갔다 하기 때문에 수신인, 제목, 날짜, 읽음 여부와는 따로 메일 내용을 크롤링
    for i in range(len(sendlist)):
        driver.find_elements_by_css_selector('strong.mail_title')[i].click()   # 해당 메일 클릭
        time.sleep(0.5)
        contents = driver.find_elements_by_id('readFrame')[0]                  # 메일 내용 크롤링
        time.sleep(0.5)
        contents_list.append(contents.text)                                      
        driver.execute_script("window.history.go(-1)")                         # 해당 메일 페이지에서 메일함으로 이동
        time.sleep(0.5)
        
        
    # 넘길 페이지 선정         개인의 마지막 페이지 설정 필요 *
    if k==10:   # 10페이지에 도달하면 다음 페이지로 넘기기
        page_bar = list(driver.find_elements_by_css_selector("div.paginate > *"))[-2]
        k=1    # 페이지가 넘어가면 자동으로 1페이지부터 시작.
    else:
        page_bar = list(driver.find_elements_by_css_selector("span.paging_numbers > *"))[k]
        k+=1
    # 페이지 넘기기
    time.sleep(0.5)
    page_bar.send_keys(Keys.ENTER)
    time.sleep(0.5)

# 데이터 프레임
df=pd.DataFrame({"FROM":data[0],"TEXT":data[1],"DATE":data[2],"CONTENTS": contents_list, "JUDGE":data[3]})
df.to_csv("naver_mail_13.csv",index=False)


# In[50]:


# 데이터 불러오기
df=pd.read_csv("naver_mail_13.csv")
df


# #### 데이터 전처리

# In[51]:


df["DATE"][:779]=df["DATE"][:779].apply(lambda x: "2022-"+x)
df["DATE"]=pd.to_datetime(df["DATE"])
df["YEAR"]=df["DATE"].dt.year     # 년도, 월, 시각으로 처리
df["MONTH"]=df["DATE"].dt.month
df["HOUR"]=df["DATE"].dt.hour
del df["DATE"]


# In[52]:


def label(x):    # 읽은 메일이라면 1, 읽지 않았다면 0
    if x=="읽은 메일":
        return 1
    else:
        return 0
df["JUDGE"]=df["JUDGE"].apply(label)


# In[53]:


df["JUDGE"].value_counts()


# - 메일 읽음 여부 (정답 라벨) imbalanced data

# In[54]:


# 메일 내용 전처리
df['CONTENTS'] = df['CONTENTS'].fillna(df['TEXT'])
# 메일 내용이 없고, 이미지만 있을 경우 결측값이 나와 메일 제목으로 내용 대체
df["CONTENTS"]=df["CONTENTS"].apply(lambda x: x.replace("\n"," "))


# In[55]:


df["from_len"]=df["FROM"].apply(lambda x : len(x)) # 수신인의 길이 피쳐로 활용
df["text_len"]=df["TEXT"].apply(lambda x : len(x)) # 제목의 길이 피쳐로 활용
df["content_len"]=df["CONTENTS"].apply(lambda x : len(x)) # 메일 내용의 길이 피쳐로 홯용


# In[56]:


df.head()


# In[57]:


# 수신인 정보 자체로만은 텍스트 분석하기에 텍스트가 부족하다고 판단하여, 제목 텍스트와 결합
df["TEXT"] = df["FROM"]+" " + df["TEXT"]


# In[58]:


#토큰화
def tokenize_kor(sent):
    return [word for word in okt.nouns(sent) if len(word) >= 2]

def tokenize_eng(sent):
    is_noun = lambda pos : pos[:2]=="NN"
    tokenized = nltk.word_tokenize(sent)
    return [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    
tokenized_doc = df['CONTENTS'].apply(lambda x: tokenize_eng(x) if x[0].isalpha() else  tokenize_kor(x)) # 토큰화
tokenized_doc_1 = df['TEXT'].apply(lambda x: tokenize_eng(x) if x[0].isalpha() else  tokenize_kor(x)) # 토큰화


# - 한글과 영어를 구분하여 각각 명사 토큰화

# In[59]:


print(tokenized_doc[0])


# In[60]:


print(tokenized_doc[4])


# In[61]:


# 역토큰화 (토큰화 작업을 역으로 되돌림)
detokenized_doc = []
for i in range(len(df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

df['CONTENTS'] = detokenized_doc


# In[62]:


df['CONTENTS'][0]


# In[63]:


# 한 글자 토큰들은 삭제 진행    -  의미없는 단어들이 포함된 경우가 많음.
def lst_check(x):
    for num, i in enumerate(x):
        if len(i)==1:
            x.pop(num)
    return x

tokenized_doc=tokenized_doc.apply(lambda x: lst_check(x))
tokenized_doc_1=tokenized_doc_1.apply(lambda x: lst_check(x))


# In[64]:


# documents을 받아서 문서 단어 행렬 만듬
def build_doc_term_mat(documents): 
    
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(document) for document in documents]

    return corpus, dictionary


# In[65]:


corpus, dictionary = build_doc_term_mat(tokenized_doc)
corpus_1, dictionary_1 = build_doc_term_mat(tokenized_doc_1)


# In[66]:


print(dictionary.token2id)


# In[67]:


lst=[]
for i in [3,5,7,10,12,15,20,30,50]:
    lda_model =models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary, alpha=1)
    lst.append((i, lda_model.log_perplexity(corpus)))


# In[68]:


lst


# In[69]:


MA_KEY = "body_ma" #내용에 대한 부분은 bady_ma 키에 적혀져 있음
NUM_TOPICS = 10     #토픽의 개수
NUM_TOPIC_WORDS = 20   #하나의 토픽에 포함되는 단어수


# In[70]:


lda_model =models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS,id2word=dictionary, alpha=1)
lda_model_1 =models.ldamodel.LdaModel(corpus_1, num_topics=NUM_TOPICS,id2word=dictionary_1, alpha=1)
# corpus와 dictionary를 파라미터로 넣고, 하이퍼파라미터 알파, 베타가 있는데 베타는 기본값, 알파는 1을 넣어본다.
# 토픽의 개수는 NUM_TOPICS=10로 지정


# In[71]:


print_term_topics('쿠폰', dictionary, lda_model)  # 특정 용어의 토픽별 가중치 출력
# 쿠폰이라는 단어는 9번 토픽에서는 0.052721의 가중치를 갖는다..


# In[72]:


# ★ 전체 문서별 주요 토픽 출력 
print_doc_topics(lda_model, corpus)                      


# In[73]:


# ★ 전체 문서별 주요 토픽 출력 
print_doc_topics(lda_model_1, corpus)                      


# In[74]:


topic_list=[]        # 문서 별 토픽 구분
topic_list_1=[]   
def what_doc_topics(model, corpus, lst):
    for doc_num, doc in enumerate(corpus):
        topic_probs = model[doc]
        topic_probs=sorted(topic_probs, key= lambda x: x[1])[-1][0]
        lst.append(topic_probs)
what_doc_topics(lda_model, corpus, topic_list)   
print(topic_list[:5])

what_doc_topics(lda_model, corpus_1, topic_list_1)   


# In[75]:


# 결과 0.nmh/topic_vis.html          # 토픽 시각화
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)

pyLDAvis.save_html(vis, "RESULT_5.html")


# ![image.png](attachment:image.png)

# In[76]:


# 토픽 분류 데이터 프레임에 추가
df["text_TOPIC"] = topic_list_1
df["content_TOPIC"] = topic_list


# In[77]:


tfidv = TfidfVectorizer(tokenizer=str.split) # tfidf 화
tfidv_from = tfidv.fit_transform(df["FROM"]).toarray()
tfidv_text = tfidv.fit_transform(df["TEXT"]).toarray()
tfidv_content = tfidv.fit_transform(df["CONTENTS"]).toarray()
print(np.shape(tfidv_from))
print(np.shape(tfidv_text))
print(np.shape(tfidv_content))


# In[78]:


#데이터 프레임 최종 완성
tfidv_from=pd.DataFrame(tfidv_from)
tfidv_text=pd.DataFrame(tfidv_text)
tfidv_text.columns=range(424,13012) # 피쳐명 중복 방지
#tfidv_content=pd.DataFrame(tfidv_content)
#tfidv_content.columns=range(13012,87977) # 피쳐명 중복 방지
df=pd.concat([df,pd.concat([pd.DataFrame(tfidv_from),pd.DataFrame(tfidv_text)],axis=1)],axis=1)


# In[79]:


del df["FROM"], df["TEXT"], df["CONTENTS"]


# In[80]:


df   # 최종 데이터 프레임 완성


# - 첫 번째 열 정답 레이블 읽음 여부
# - 2 ~ 4번 째 열, 시각 관련 수신 날짜 테이블
# - 5 ~ 7번 째 열, 각각의 텍스트 길이
# - 8 ~ 9번 째 열, 발신인 + 메일 제목 토픽, 메일 내용 토픽
# - 10번 째 열, 메일 제목 tf-idf ( 메일 내용 th-idf는 
# 

# In[81]:


# 정답 레이블 분리
y_train = df["JUDGE"]
del df["JUDGE"]


# #### imbalanced_data

# In[82]:


# imbalanced data로 인해 전부 0으로 예측하기만 해도 손쉽게 정확도가 90프로 이상
# -> class 1 (읽을 것으로 예측되는 데이터) 를 잘 맞출 수 있도록 데이터 리샘플링 필요
# undersampling 기법, oversampling 기법 사용 후 class1 에 대한 recall 이 더 높은 경우 채택
def imbalanced_process(model,x,y,z=0):
    X_train, X_dev, y_train, y_dev = train_test_split(x,y, test_size=0.3, random_state=77) 
    if z!=0:
        X, y = z.fit_resample(X_train, y_train)
    else: X,y=x,y
    model.fit(X_train, y_train)
    y_pred=model.predict(X_train)
    y1_pred=model.predict(X_dev)
    print("TRAIN"); print(classification_report(y_train, y_pred, target_names=['class 0', 'class 1']))
    print("VALID"); print(classification_report(y_dev, y1_pred, target_names=['class 0', 'class 1']))
model=LGBMClassifier(random_state=0, n_jobs=-1, class_weight={1:10000,0:1})


# In[83]:


# 데이터 리샘플링 이전
imbalanced_process(model,df,y_train)


# -> accuracy가 93%로 매우 높게 나오지만 class 1에 대한 recall 점수는 상당히 낮음
# 
# - 지금 만들고자 하는 시스템은 메일이 왔을 경우, 내가 읽을 것 같은 메일을 예측하여 추천하는 시스템이다.
# 
# - 읽지 않아도 되는 메일을 추천하는 경우는 어느정도 괜찮지만, 읽어야 할 메일을 예측하지 못할 경우 더 큰 문제를 야기한다.
# 
# - 따라서 valid 데이터에서 class1의 recall 점수와 f1-score 를 이용하여 RESAMPLING 기법 채택

# In[84]:


# Method: Using SHAP values 
# DF, based on which importance is checked
X_importance = df

# Explain model predictions using shap library:
model = LGBMClassifier(random_state=0, n_jobs=-1, class_weight={1:10000,0:1}).fit(df, y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_importance)

# Plot summary_plot as barplot:
plt.title("피쳐 중요도")
shap.summary_plot(shap_values, X_importance, plot_type='bar')


# In[85]:


shap_sum = np.abs(shap_values).mean(axis=1)[1,:]
importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df


# In[86]:


# feature 중요도가 0 이상
SHAP_THRESHOLD = 0
features_selected = importance_df.query('shap_importance > @SHAP_THRESHOLD').column_name.tolist()
df = df[features_selected]
print(df.shape)


# In[87]:


gc.collect()   # 가비지 콜렉터를 이용해 메모리 절약


# In[88]:


print("데이터 언더샘플링 RandomUnderSampler")
imbalanced_process(model,df,y_train,RandomUnderSampler(random_state=0))

print("데이터 언더샘플링 NearMiss")
imbalanced_process(model,df,y_train,NearMiss(version=1,n_jobs=-1))

print("데이터 오버샘플링 RandomOverSampler")
imbalanced_process(model,df,y_train,RandomOverSampler(random_state=0))

print("데이터 오버샘플링 SMOTE")
imbalanced_process(model,df,y_train,SMOTE(random_state=0,k_neighbors=5,n_jobs=-1))

print("데이터 오버샘플링 BorderlineSMOTE")
imbalanced_process(model,df,y_train,BorderlineSMOTE(kind = 'borderline-1',random_state=0,n_jobs=-1))


# - 오버샘플링 RandomOverSampler 채택

# #### train, validation 데이터 분할

# In[89]:


# 검증 데이터 분리
X_train, X_dev, y_train, y_dev = train_test_split(df, y_train, test_size=0.3, random_state=77) 
left_df = len(X_train)
train, label = X_train, y_train


# In[90]:


# # 한 쪽 클래스가 20프로 이하일 경우 데이터 샘플링
# if y_train.value_counts()[0]/len(y_train) >=0.8 or y_train.value_counts()[0]/len(y_train) <= 0.2: 
#     X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)
#     print(f' 데이터 resampling 전 {left_df} 개 \n 데이터 resampling 후 {len(X_train)} 개')
# else:
#     print("resmapling 필요 없음.")


# In[91]:


gc.collect()   # 가비지 콜렉터를 이용해 메모리 절약


# #### Modeling

# - LGBM 튜닝 모델 사용
# - epoch 30

# In[92]:


pbounds = {
    'n_estimators':(50,800),
    'learning_rate':(0.001,1.5),
    'max_depth':(2, 32),
    'num_leaves':(2, 64),
    'subsample':(0.5, 0.95),
    'colsample_bytree':(0.5, 0.95),
    'max_bin':(10, 500),
    'reg_lambda':(0.001, 50),
    'reg_alpha':(0.001, 50)
}
def lgbm_opt(n_estimators, learning_rate, max_depth, num_leaves,
             subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        "n_estimators":int(round(n_estimators)), 
        "learning_rate":learning_rate,
        'max_depth':int(round(max_depth)),
        'num_leaves':int(round(num_leaves)),
        'subsample':max(min(subsample, 1), 0),
        'colsample_bytree':max(min(colsample_bytree, 1), 0),
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'max_bin':int(max_bin)
    }
    
#neg_mean_squared_error
    lgbm = LGBMClassifier(random_state=0, **params, n_jobs=-1, class_weight={1:10000,0:1})
    lgbm.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="f1", eval_set=(X_dev, y_dev), verbose=False)
    score = cross_val_score(lgbm, X_dev, y_dev, scoring="roc_auc", cv=2, n_jobs=-1)

    return np.mean(score)

BO_lgbm = BayesianOptimization(f = lgbm_opt, pbounds = pbounds, random_state=1004)
BO_lgbm.maximize(init_points=10, n_iter=20)

max_params_lgbm = BO_lgbm.max['params']
max_params_lgbm


# In[93]:


max_params_lgbm['n_estimators'] = int(max_params_lgbm['n_estimators'])
max_params_lgbm['max_depth'] = int(max_params_lgbm['max_depth'])
max_params_lgbm['max_bin'] = int(max_params_lgbm['max_bin'])
max_params_lgbm['num_leaves'] = int(max_params_lgbm['num_leaves'])
lgbm_clf = LGBMClassifier(random_state = 0, **max_params_lgbm)
lgbm_clf.fit(X_train,y_train)


# #### Scoring

# In[94]:


# train score
y_pred = lgbm_clf.predict(train)
print("Train score")
print(classification_report(label, y_pred, target_names=['class 0', 'class 1']))

# validation score
y_pred = lgbm_clf.predict(X_dev)
print("Validation score")
print(classification_report(y_dev, y_pred, target_names=['class 0', 'class 1']))


# In[95]:


model = LogisticRegression(random_state=123)
model.fit(X_train, y_train)

# train score
y_pred = model.predict(train)
print("Train score")
print(classification_report(label, y_pred, target_names=['class 0', 'class 1']))

# validation score
y_pred = model.predict(X_dev)
print("Validation score")
print(classification_report(y_dev, y_pred, target_names=['class 0', 'class 1']))


# ### 예측 결과 (성능)
# -> CLASS1측면에 대한 설명 
# - LGBM 모델 : 읽을 것이라고 예측한 것 중 내가 읽을 예정이었던 것은 5%를 차지한다.  (정밀도)
#               TRUE값 중 POSITIVE로 맞게 예측한 비율은 58%이다. (재현율)
#               읽지 않을 것(FALSE)값 중 패스할 수 있는 메일의 비율은 76%이다. 76%에 메일을 바로 보지 않고, 처리 가능. ( CLASS0의 측면 재현율) 
#               

# #### 한계점
# - 나의 메일 데이터 읽음, 안읽음 데이터는 imbalanced_data이기에 예측에 어려운 점이 있었다.
# - 크롤링을 하기 위해서 메일 마다 전부 들어갔다 나왔다 해야했기에 많은 시간 및 자원 소모가 있었고 데이터 수집 양(행)이 이전보다 줄어들어 예측 성능이 약간 감소하였다.
# - 어느정도의 사용자 기반의 메일 읽기 예측을 할 수 있으나, 성능이 만족스럽지 못하다. (개선 필요)
# - class1의 재현율을 1에 가깝게하여 중요한 메일을 놓치지 않게끔 성능향상이 필요하다.
