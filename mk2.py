import tensorflow as tf
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,LeakyReLU


#code reference:
#주가저장:devpouch.tistory.com/151
#주가예측:https://www.youtube.com/watch?v=LLLVYkXJw30
#Google Bard

# 종목 코드
ticker = "067160.KQ"

# 기간
df = yf.download(ticker, start="2020-06-30", end="2020-12-30")
df2 = yf.download(ticker, start="2022-09-27", end="2023-12-14")

# 이동 평균
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
# 이동 평균
df2['MA5'] = df2['Close'].rolling(window=5).mean()
df2['MA10'] = df2['Close'].rolling(window=10).mean()
df2['MA20'] = df2['Close'].rolling(window=20).mean()

df=df.dropna()
df=df.dropna()

# 그래프
plt.figure(figsize=(7,4))
plt.plot(df.index, df['Close'], label='Close', color='r')
plt.plot(df.index, df['MA5'], label='MA5', color='g')
plt.plot(df.index, df['MA10'], label='MA10', color='m')
plt.plot(df.index, df['MA20'], label='MA20', color='c')

plt.legend(loc='best')
plt.grid()
plt.xlabel('date')
plt.ylabel('price(won)')
plt.show()

# 날짜를 제외한 숫자로 표현되는 모든 항목을 0~1로 정규화
def normalize(df):
    for col in df.columns:
        if col != 'Date':
            if df[col].dtype == 'float' or df[col].dtype == 'int64':
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df.to_numpy()

# 정규화된 데이터를 numpy 배열로 반환
df_normalized = normalize(df)
df2_normalized = normalize(df2)



feature_cols=['MA5','MA10','MA20','Close']
label_cols=['Close']

feature_df=pd.DataFrame(df_normalized[:,0:4],columns=feature_cols)
label_df=pd.DataFrame(df_normalized[:,4],columns=label_cols)
feature_df2=pd.DataFrame(df2_normalized[:,0:4],columns=feature_cols)
label_df2=pd.DataFrame(df2_normalized[:,4],columns=label_cols)

print(feature_df)
print(label_df)

label_np=label_df.to_numpy()
feature_np=feature_df.to_numpy()
label_np2=label_df2.to_numpy()
feature_np2=feature_df2.to_numpy()

#시계열데이터생성 
def make_sequence_dataset(feature,label,window_size):
    feature_list=[]
    label_list=[]
    
    for i in range(len(feature)-window_size):
        
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])
    
    return np.array(feature_list),np.array(label_list)


  
 


#LSTM 구현
window_size=40
X,Y=make_sequence_dataset(feature_np,label_np,window_size)
X2,Y2=make_sequence_dataset(feature_np2,label_np2,window_size)
print(X.shape,Y.shape)


x_train=X
y_train=Y
x_test=X2
y_test=Y2

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)



model=Sequential()
model.add(LSTM(128,activation='LeakyReLU',input_shape=x_train[0].shape))
model.add(Dense(1,activation='linear'))
model.add(LeakyReLU())
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
learning_rate=0.0000001
model.compile(loss='mse',optimizer='adam',metrics=['accuracy','mae'])
early_stop=EarlyStopping(monitor='val_loss',patience=5)
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=12,callbacks=[early_stop])

#그래프
pred=model.predict(x_test)
plt.figure(figsize=(7,4))
plt.title(f'5,10,20,Close, windowsize={window_size}')
plt.ylabel('close')
plt.xlabel('period')
plt.plot(y_test,label='actual')
plt.plot(pred,label='prediction')
plt.grid()
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid()
plt.show()












