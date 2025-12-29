import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import pipeline

print("⏳ Loading FinBERT...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except:
    sentiment_pipeline = None

class StockAnalystEngine:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = self.stock.info
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = None
        self.time_step = 100 

    def get_data(self):
        df = self.stock.history(period="2y")
        return df if not df.empty else None

    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    def train_stacked_lstm(self, df):
        data = df.filter(['Close']).values
        self.scaled_data = self.scaler.fit_transform(data)
        
        training_size = int(len(self.scaled_data) * 0.65)
        train_data = self.scaled_data[0:training_size,:]
        
        X_train, y_train = self.create_dataset(train_data, self.time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, verbose=0)
        
        self.model = model
        return model

    def predict_future_30_days(self):
        if not self.model: return []
        
        temp_input = self.scaled_data[-self.time_step:].tolist()
        temp_input = [item for sublist in temp_input for item in sublist]
        
        lst_output = []
        n_steps = 100
        i = 0
        
        while(i < 30):
            if(len(temp_input) > 100):
                x_input = np.array(temp_input[1:]).reshape(1, n_steps, 1)
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i += 1
            else:
                x_input = np.array(temp_input).reshape((1, n_steps, 1))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i += 1
        
        return self.scaler.inverse_transform(lst_output)

    def get_fundamentals(self):
        """Calculates a Fundamental Score (0 to 1)"""
        try:
            pe = self.info.get('trailingPE', 25) 
            peg = self.info.get('pegRatio', 1.5)
            growth = self.info.get('revenueGrowth', 0)
            target = self.info.get('targetMeanPrice', 0)
            current = self.info.get('currentPrice', 1)

            score = 0.5 
            
            if pe < 20: score += 0.1
            if peg < 1: score += 0.1
            
            if growth > 0.10: score += 0.1
            
            if target > current: score += 0.1
            
            return min(max(score, 0), 1) 
        except: return 0.5

    def analyze_news(self):
        try:
            news = self.stock.news
            if not news: return 0, []
            scores, headlines = [], []
            for n in news[:5]:
                title = n.get('title', '')
                headlines.append(title)
                if sentiment_pipeline:
                    res = sentiment_pipeline(title)[0]
                    sc = res['score'] if res['label'] == 'positive' else -res['score']
                    scores.append(sc)
            return round(np.mean(scores), 2) if scores else 0, headlines
        except: return 0, []

    def fine_tune_forecast(self, raw_prices, fund_score, sent_score):
        """Adjusts the blind LSTM curve based on Reality"""
        adjusted_prices = []
        
        fund_bias = (fund_score - 0.5) * 0.10 
        
        sent_bias = sent_score * 0.05
        
        total_bias = fund_bias + sent_bias
        
        for i, price in enumerate(raw_prices):
            time_weight = (i + 1) / 30 
            
            adj_price = price[0] * (1 + (total_bias * time_weight))
            adjusted_prices.append(round(adj_price, 2))
            
        return adjusted_prices