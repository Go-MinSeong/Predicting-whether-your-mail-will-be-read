#!/usr/bin/env python
# coding: utf-8

# ## **사용자 기반 네이버 메일 알림 시스템**
# 
# [1. 네이버 메일 자동 로그인](#네이버-메일-자동-로그인)
# 
# [ 2. 사용자의 메일 데이터를 수집](#사용자의-메일-데이터를-수집) <br/>
#     (읽음 여부, 발신인, 메일 제목, 발신 날짜)  <br/>
#     메일 5000개 당 약 한 시간 소요
#     
#     
# [ 3. 데이터 전처리](#데이터-전처리) <br/>
#     읽음 여부는 읽은 메일을 1, 읽지 않은 경우 0 <br/>
#     발신인과 메일 제목은 명사 추출 후 이를 활용하여 TF-IDF 벡터화 진행 <br/>
#     발신인 메일 제목의 길이를 활용하여 피쳐 사용 <br/>
#     발신 날짜는 해당 년도, 월, 시각 이용
# 
# [ 4. imbalanced_data 확인 여부 후, 데이터 resampling ](#imbalanced_data)
# 
# [ 5. train, validation 데이터 분할](#train,-validation-데이터-분할)
# 
# [ 6. Features selection](#Features-selection)
# 
# [ 7. Modeling ( LGBM Classifier, Logistic Regression 사용 )](#Modeling)
# 
# [ 8. Scoring](#Scoring)
# 
# [ 9. 사용자 메일 내용 워드 클라우딩](#사용자-메일-내용-워드-클라우딩)

# In[1]:


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
from konlpy.tag import Okt
from collections import Counter

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


# #### 네이버 메일 자동 로그인
# - 자신의 아이디와 비밀번호를 해당 칸에 입력해야 로그인 가능

# In[3]:


# 드라이버 설정
driver = webdriver.Chrome('C:\chromedriver.exe')
time.sleep(0.5)
driver.implicitly_wait(3)

# 페이지 설정
driver.get('https://mail.naver.com')

# 네이버 자동 로그인
driver.find_element_by_name('id').click()

# 자동 로그인 우회 방식
pyperclip.copy('아이디입력') # 아이디
driver.find_element_by_name('id').send_keys(Keys.CONTROL, 'v')
driver.find_element_by_name('pw').click() # 패스워드
pyperclip.copy('비밀번호입력')
driver.find_element_by_name('pw').send_keys(Keys.CONTROL, 'v')

# 로그인 클릭
driver.find_element_by_id('log.login').click()

html = driver.page_source 
soup = BeautifulSoup(html, 'html.parser')

# 팝업창 닫기
driver.find_element_by_xpath("//*[@id='smartFolderLayer']/div/div/button").send_keys(Keys.ENTER)


# #### 사용자의 메일 데이터를 수집
# - 메일함 데이터 수집 (동적 크롤링) 5000개당 약 한시간 소요

# In[ ]:


# 페이지 넘기기
k=1
data=[[] for _ in range(4)]
for _ in tqdm(range(1011)):  # 1011 페이지까지 있음.  # 2시간 소요 
    # 넘길 페이지 선정         개인의 마지막 페이지 설정 필요 *
    if k==10:   # 10페이지에 도달하면 다음 페이지로 넘기기
        page_bar = list(driver.find_elements_by_css_selector("div.paginate > *"))[-2]
        k=1    # 페이지가 넘어가면 자동으로 1페이지부터 시작.
    else:
        page_bar = list(driver.find_elements_by_css_selector("span.paging_numbers > *"))[k]
        k+=1
    
    # 페이지가 넘어갈 때마다 크롤링 재설정
    sendlist = driver.find_elements_by_css_selector('div.name')
    time.sleep(1)
    maillist = driver.find_elements_by_css_selector('strong.mail_title')
    time.sleep(1)
    read_jugements = driver.find_elements_by_css_selector('li.read > a')
    time.sleep(1)
    dates = driver.find_elements_by_css_selector('li.iDate')
    time.sleep(1)
    # 데이터 수집     발신인 , 제목
    
    for mail in zip(sendlist, maillist, dates, read_jugements):
        for j in range(3):
            data[j].append(mail[j].text)
        data[3].append(mail[3].get_attribute("title"))
    time.sleep(1)
    
    # 페이지 넘기기
    page_bar.send_keys(Keys.ENTER)
    time.sleep(1)

# 데이터 프레임
df=pd.DataFrame({"FROM":data[0],"TEXT":data[1],"DATE":data[2],"JUDGE":data[3]})
df.to_csv("naver_mail",index=False)


# In[2]:


# 데이터 불러오기
df=pd.read_csv("naver_mail")
df.head()


# #### 데이터 전처리

# In[3]:


df["DATE"].iloc[0]="04-18 19:30"   # 날짜 피쳐 전처리
df["DATE"][:682]=df["DATE"][:682].apply(lambda x: "2022-"+x)
df["DATE"]=pd.to_datetime(df["DATE"])
df["YEAR"]=df["DATE"].dt.year     # 년도, 월, 시각으로 처리
df["MONTH"]=df["DATE"].dt.month
df["HOUR"]=df["DATE"].dt.hour
del df["DATE"]


# In[4]:


def label(x):    # 읽은 메일이라면 1, 읽지 않았다면 0
    if x=="읽은 메일":
        return 1
    else:
        return 0
df["JUDGE"]=df["JUDGE"].apply(label)


# - 메일 읽음 여부 (정답 라벨) imbalanced data

# In[5]:


df["JUDGE"].value_counts()


# In[6]:


df["from_len"]=df["FROM"].apply(lambda x : len(x)) # 텍스트의 길이 피쳐로 활용
df["text_len"]=df["TEXT"].apply(lambda x : len(x))


# In[7]:


# 내용에서 명사만 추출해서 사용
okt = Okt()
text=' '.join(df["TEXT"]) # 뒤에서 메일 자기 분석시 사용
df["TEXT"]=df["TEXT"].apply(lambda x: ','.join(okt.nouns(x))) # 명사만 사용
df["TEXT"]


# In[8]:


tfidv = TfidfVectorizer(tokenizer=str.split) # tfidf 화
tfidv_from = tfidv.fit_transform(df["FROM"]).toarray()
tfidv_text = tfidv.fit_transform(df["TEXT"]).toarray()
print(np.shape(tfidv_from))
print(np.shape(tfidv_text))
del df["FROM"],df["TEXT"]


# In[9]:


#데이터 프레임 최종 완성
tfidv_from=pd.DataFrame(tfidv_from)
tfidv_text=pd.DataFrame(tfidv_text)
tfidv_text.columns=range(938,11078) # 피쳐명 중복 방지
df=pd.concat([df,pd.concat([pd.DataFrame(tfidv_from),pd.DataFrame(tfidv_text)],axis=1)],axis=1)


# #### imbalanced_data

# In[10]:


# imbalanced data로 인해 전부 0으로 예측하기만 해도 손쉽게 정확도가 90프로 이상
# -> class 1 (읽을 것으로 예측되는 데이터) 를 잘 맞출 수 있도록 데이터 리샘플링 필요
# undersampling 기법, oversampling 기법 사용 후 class1 에 대한 recall 이 더 높은 경우 채택
def imbalanced_process(model,x,y,z=0):
    X_train, X_dev, y_train, y_dev = train_test_split(x,y, test_size=0.3, random_state=77) 
    if z!=0:
        X, y = z.fit_resample(X_train, y_train)
    else: X,y=x,y
    model.fit(X, y)
    y_pred=model.predict(X_train)
    y1_pred=model.predict(X_dev)
    print("TRAIN"); print(classification_report(y_train, y_pred, target_names=['class 0', 'class 1']))
    print("VALID"); print(classification_report(y_dev, y1_pred, target_names=['class 0', 'class 1']))
model=LGBMClassifier(random_state=0, n_jobs=-1)


# In[11]:


# 정답 레이블 분리
y_train = df["JUDGE"]
del df["JUDGE"]


# In[12]:


# 데이터 리샘플링 이전
imbalanced_process(model,df,y_train)


# -> accuracy가 98%로 매우 높게 나오지만 class 1에 대한 recall 점수는 상당히 낮음
# 
# - 지금 만들고자 하는 시스템은 메일이 왔을 경우, 내가 읽을 것 같은 메일을 예측하여 추천하는 시스템이다.
# 
# - 읽지 않아도 되는 메일을 추천하는 경우는 어느정도 괜찮지만, 읽어야 할 메일을 예측하지 못할 경우 더 큰 문제를 야기한다.
# 
# - 따라서 valid 데이터에서 class1의 recall 점수와 f1-score 를 이용하여 RESAMPLING 기법 채택

# In[13]:


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

# In[14]:


# 검증 데이터 분리
X_train, X_dev, y_train, y_dev = train_test_split(df, y_train, test_size=0.3, random_state=77) 
left_df = len(X_train)
train, label = X_train, y_train


# In[24]:


# 한 쪽 클래스가 20프로 이하일 경우 데이터 샘플링
if y_train.value_counts()[0]/len(y_train) >=0.8 or y_train.value_counts()[0]/len(y_train) <= 0.2: 
    X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)
    print(f' 데이터 resampling 전 {left_df} 개 \n 데이터 resampling 후 {len(X_train)} 개')
else:
    print("resmapling 필요 없음.")


# ##### Features selection

# In[25]:


# Method: Using SHAP values 
# DF, based on which importance is checked
X_importance = X_train

# Explain model predictions using shap library:
model = LGBMClassifier(random_state=0, n_jobs=-1).fit(X_train, y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_importance)

# Plot summary_plot as barplot:
plt.title("피쳐 중요도")
shap.summary_plot(shap_values, X_importance, plot_type='bar')


# In[26]:


shap_sum = np.abs(shap_values).mean(axis=1)[1,:]
importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df


# In[27]:


# feature 중요도가 0 이상
SHAP_THRESHOLD = 0
features_selected = importance_df.query('shap_importance > @SHAP_THRESHOLD').column_name.tolist()
x_train = X_train[features_selected]
x_dev =X_dev[features_selected]
train =train[features_selected]
print(x_train.shape)


# In[28]:


gc.collect()   # 가비지 콜렉터를 이용해 메모리 절약


# #### Modeling

# - LGBM 튜닝 모델 사용
# - epoch 30

# In[45]:


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
    lgbm = LGBMClassifier(random_state=0, **params, n_jobs=-1)
    lgbm.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="f1", eval_set=(x_dev, y_dev), verbose=False)
    score = cross_val_score(lgbm, x_dev, y_dev, scoring="roc_auc", cv=2, n_jobs=-1)

    return np.mean(score)

BO_lgbm = BayesianOptimization(f = lgbm_opt, pbounds = pbounds, random_state=1004)
BO_lgbm.maximize(init_points=10, n_iter=20)

max_params_lgbm = BO_lgbm.max['params']
max_params_lgbm


# In[46]:


max_params_lgbm['n_estimators'] = int(max_params_lgbm['n_estimators'])
max_params_lgbm['max_depth'] = int(max_params_lgbm['max_depth'])
max_params_lgbm['max_bin'] = int(max_params_lgbm['max_bin'])
max_params_lgbm['num_leaves'] = int(max_params_lgbm['num_leaves'])
lgbm_clf = LGBMClassifier(random_state = 0, **max_params_lgbm)
lgbm_clf.fit(x_train,y_train)


# #### Scoring

# In[47]:


# train score
y_pred = lgbm_clf.predict(train)
print("Train score")
print(classification_report(label, y_pred, target_names=['class 0', 'class 1']))

# validation score
y_pred = lgbm_clf.predict(x_dev)
print("Validation score")
print(classification_report(y_dev, y_pred, target_names=['class 0', 'class 1']))


# In[32]:


model = LogisticRegression(random_state=123)
model.fit(x_train, y_train)

# train score
y_pred = model.predict(train)
print("Train score")
print(classification_report(label, y_pred, target_names=['class 0', 'class 1']))

# validation score
y_pred = model.predict(x_dev)
print("Validation score")
print(classification_report(y_dev, y_pred, target_names=['class 0', 'class 1']))


# ### 예측 결과 (성능)
# -> CLASS1측면에 대한 설명   ( 읽을 것이다. )
# - LGBM 모델 : 읽을 것이라고 예측한 것 중 내가 읽을 예정이었던 것은 14%를 차지한다.  (정밀도)
#               TRUE값 중 POSITIVE로 맞게 예측한 비율은 53%이다. (재현율)
#               읽지 않을 것(FALSE)값 중 패스할 수 있는 메일의 비율은 93%이다. 93%에 메일을 바로 보지 않고, 처리 가능. ( CLASS0의 측면 재현율) 
#               
# - LOGISTIC REGRESSION 모델 : 읽을 것이라고 예측한 것 중 내가 읽을 예정이었던 것은 6%를 차지한다. (정밀도)
#                              TRUE값 중 POSITIVE로 맞게 예측한 비율은 85%이다.    (재현율)
#                              읽지 않을 것(FALSE)값 중 패스할 수 있는 메일의 비율은 69%이다. 69%에 메일을 바로 보지 않고, 처리 가능. ( CLASS0의 측면 재현율)
#                              
#                          

# #### 한계점
# - 나의 메일 데이터 읽음, 안읽음 데이터는 imbalanced_data이기에 예측에 어려운 점이 있었다.
# - 어느정도의 사용자 기반의 메일 읽기 예측을 할 수 있으나, 성능이 만족스럽지 못하다. (개선 필요)
# - 메일의 피쳐 중 읽음 여부, 발신인, 메일 제목, 날짜만 사용하였으나 더 많은 피쳐를 사용할 필요가 있다.
# - 머신러닝 기반의 모델만을 사용하여 딥러닝 기반 모델시 성능이 올라갈 수도 있다.
# - class1의 재현율을 1에 가깝게하여 중요한 메일을 놓치지 않게끔 성능향상이 필요하다.

# #### 사용자 메일 내용 워드 클라우딩
# - 위와 별개 내용
# - 추가적 코드 작성

# In[33]:


nouns = okt.nouns(text)
words = [n for n in nouns if len(n) > 1] 
plt.figure(figsize=(20,10))
words = nltk.Text(words)
plt.title("메일 내용 분석")
plt.xticks(rotation=50)
words.plot(50) # 50개만
plt.show()

# 단어 빈도표를 보고 일부 단어 불용어 처리
stopword = ['광고']
text = [i for i in words if i not in stopword]
text_cnt = Counter(text)

# colormap 글씨색 설정 가능, img_mask, mask 모양 설정
img_mask = np.array(Image.open("man.png"))    # 사람 이미지

wordcloud = WordCloud(font_path = 'NanumGothic.otf',   # 나눔고딕 폰트
                      background_color='white',
                      colormap = "Accent_r", 
                      mask = img_mask, 
                      random_state = 20, 
                      max_words = 500).generate_from_frequencies(text_cnt)

plt.figure(figsize = (20, 10))
plt.imshow(wordcloud, interpolation = "bilinear")        
plt.axis("off")
plt.savefig("Minseong.png")
plt.show()


# #### 메일 제목을 이용한 자신의 메일함 워드 클라우딩 및 그래프 표현
# - 광고, 안내, 무료, 할인, 추가 등 많은 광고 어휘 데이터가 차지하고 있어, 대부분 광고 메일임을 확인할 수 있다.
# - 확인을 요하는 메일, 네이버 측에서 전달하는 메일이 많으며 웹툰 관련 메일 또한 많다.
# - 전체적으로 다양한 문자의 워드 클라우딩이 형성되고 있다.
