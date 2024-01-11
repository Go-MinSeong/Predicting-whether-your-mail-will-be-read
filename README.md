<img width="400" alt="image" src="https://github.com/Go-MinSeong/Predicting-whether-your-mail-will-be-read/assets/91547241/f0c550ca-8f59-46ec-817f-ad03ebbd9f89"># Predicting-whether-your-mail-will-be-read

## 프로젝트 기간
2022.04 ~ 2022.06 ( 텍스트 데이터 분석 )

## 프로젝트 계기
현대 사회에서 하루마다 많은 메일 수신이 온다. 하지만 많은 메일 중 나에게 중요한 메일도 있지만, 광고성 메일, 인증 메일, 알림 메일 등 많은 불필요한 메일이 존재하며 우리는 이 중 나에게 필요한 메일을 확인하기 위해서는 불필요한 시간이 소모 된다.



** 따라서 자신의 메일함 데이터를 이용한, 내가 읽을 것 같은 메일을 예측하여 알려주는 시스템을 구현하고자 한다. **



## 목표


중요한 메일이 왔을 경우, 확인을 할 수 있도록 읽어야 하는 메일에 대한 높은 성능을 만들어야 한다.

중요하지 않은 메일을 건너 뛸 수 있도록 잘 걸러내야 한다.

**자신의 메일함을 기반으로 분석하기 때문에 보다 자신에게 적합한 사용자 기반 메일 분류기를 만들어내도록 한다.**

## **사용자 기반 네이버 메일 알림 시스템 방법론**

1. 네이버 메일 자동 로그인


2. 크롤링을 통한 사용자의 메일 데이터를 수집 <br/>
    (읽음 여부, 발신인, 메일 제목, 발신 날짜)  <br/>
    메일 5000개 당 약 한 시간 소요
    
    
3. 데이터 전처리 <br/>
    읽음 여부는 읽은 메일을 1, 읽지 않은 경우 0 <br/>
    발신인과 메일 제목은 명사 추출 후 이를 활용하여 TF-IDF 벡터화 진행 <br/>
    발신인 메일 제목의 길이를 활용하여 피쳐 사용 <br/>
    발신 날짜는 해당 년도, 월, 시각 이용


4. imbalanced_data 확인 여부 후, 데이터 resampling 


5. train, validation 데이터 분할


6. Features selection


7. Modeling ( LGBM Classifier, Logistic Regression 사용 )


8. Scoring


9. 사용자 메일 내용 워드 클라우딩


[메일 제목 기반 워드클라우드]![Minseong](https://user-images.githubusercontent.com/91547241/216985220-55ca6a36-28ee-4b23-9171-21a5948fb749.png)


[메일 제목 기반 워드클라우드 2]![Minseong2](https://user-images.githubusercontent.com/91547241/216985246-1864fec1-c4d9-45dd-bb12-bcc2f5eb23f4.png)



## 평가 방법


Unbalanced data이므로, recall과 precision을 score로 산정



Header|precision|recall|f1-score|support
---|---|---|---|---|
읽지 않은 메일|1.00|0.69|0.81|4451|
읽은 메일|0.06|0.85|0.11|99|


## Result

실제로 수신 확인한 데이터 중 수신 확인할 것이라 예측한 비율 85%
실제로 수신 확인하지 않은 데이터 중 수신 확인하지 않을 것이라 예측한 비율 69%

아직, 서비스를 위한 성능으로는 부족하다고 느껴지지만, 부족한 데이터 추가적인 피쳐와 모델링을 사용한다면 성능을 더 높일 수 있을 것이라 기대한다.
