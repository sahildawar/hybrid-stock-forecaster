
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine import StockAnalystEngine

os.environ["GEMINI_API_KEY"] = "YOUR_KEY_HERE"

def configure_gemini():
    """
    Auto-discovers the best working model for your specific API Key.
    No more guessing names.
    """
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
        print("🔍 Scanning available models for your API key...")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        print(f"📋 Found models: {available_models}")

        priority_list = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro",
            "models/gemini-1.0-pro"
        ]

        for target in priority_list:
            if target in available_models:
                print(f"✅ Selected Best Model: {target}")
                return genai.GenerativeModel(target)
        
        if available_models:
            first_model = available_models[0]
            print(f"⚠️ specific models not found, falling back to: {first_model}")
            return genai.GenerativeModel(first_model)
            
        print("⚠️ Model scan failed. Forcing 'gemini-1.5-flash'...")
        return genai.GenerativeModel("gemini-1.5-flash")

    except Exception as e:
        print(f"❌ Critical Error in Model Setup: {e}")
        return None

model = configure_gemini()
app = FastAPI(title="Hedge Fund AI Agent")

class Request(BaseModel):
    ticker: str

@app.post("/analyze")
def analyze(req: Request):
    if not model:
        raise HTTPException(500, "Gemini API failed to initialize. Check console logs.")

    try:
        engine = StockAnalystEngine(req.ticker)
        df = engine.get_data()
        if df is None: raise HTTPException(404, detail="Ticker not found")
        
        engine.train_stacked_lstm(df)
        raw_forecast = engine.predict_future_30_days()
        
        fund_score = engine.get_fundamentals()
        sent_score, headlines = engine.analyze_news()
        
        tuned_forecast = engine.fine_tune_forecast(raw_forecast, fund_score, sent_score)
        
        current_price = round(df['Close'].iloc[-1], 2)
        
        prompt = f"""
        Role: Senior Portfolio Manager.
        Asset: {req.ticker}
        
        --- DATA ---
        Current Price: ${current_price}
        Fundamental Score: {fund_score}/1.0
        News Sentiment: {sent_score}
        
        --- FORECASTS ---
        Raw LSTM (Math only): ${raw_forecast[-1][0]:.2f}
        Fine-Tuned (Reality): ${tuned_forecast[-1]}
        
        Task:
        1. Compare Math vs. Reality.
        2. Give a 30-day trading strategy.
        """
        
        response = model.generate_content(prompt)
        
        return {
            "symbol": req.ticker,
            "current_price": current_price,
            "raw_forecast": [round(x[0], 2) for x in raw_forecast],
            "tuned_forecast": tuned_forecast,
            "fund_score": fund_score,
            "sent_score": sent_score,
            "headlines": headlines,
            "report": response.text
        }
        
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)