# 📊 Stock Insight Multi-Agent App

An intelligent financial assistant that combines **LLMs**, **LangChain**, **Streamlit**, and powerful real-time APIs to deliver actionable stock insights. This multi-agent app acts like a **personal stock analyst**, capable of understanding financial queries, routing them to specialized agents, and responding with structured, up-to-date insights.

Whether you're an investor, analyst, or market enthusiast, this app simplifies research and decision-making by blending:

- 🔍 **Live financial data**
- 🧠 **AI-driven analysis**
- 💬 **Conversational interface**

Whether you're an investor, a financial researcher, or just curious about market trends, this app acts like a **personal stock analyst**, combining:

## 💼 What It Can Do

- Track **real-time stock prices** (intraday & daily)
- Retrieve **fundamentals** and **financial statements**
- Analyze **technical indicators** (like RSI)
- Generate **buy/hold/sell** recommendations
- Detect **market sentiment** from news sources
- Extract **insights from the web**
- Handle **complex financial queries** through agent collaboration

---

## 🚀 Features

### 📈 Stock Data & Analysis
- Intraday prices, daily trends, and company fundamentals via **Alpha Vantage**
- RSI and MACD indicators for technical analysis
- Financial reports like **income statements**
- Agent-powered **buy/hold/sell** logic based on P/E, RSI, and sentiment

### 📰 Sentiment Analysis
- Recent news headlines and sentiment via **Alpha Vantage** and **NewsAPI**
- Summary of tone and implications from financial media

### 🌐 Insights & RAG
- Web search using **Tavily API**
- Content extraction using **BeautifulSoup**
- Embedded similarity search with **FAISS** and **Sentence-BERT**
- Retrieve top web insights and summarize them intelligently

### 🤖 Multi-Agent LLM System
Specialized agents using **LangChain Tool Calling**:
- `Stock Data Agent`: Prices, indicators, fundamentals, recommendations
- `Sentiment Agent`: News sentiment and trends
- `Insights Agent`: Web search and summarization
- `General Agent`: Handles any query using Tavily search

### 🧠 Smart Query Classification
Automatically detects whether your query is:
- Stock-specific
- Sentiment-oriented
- General (or multiple)
Routes it to the right agents for a coordinated response.

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Alpha Vantage API](https://www.alphavantage.co/)
- [NewsAPI](https://newsapi.org/)
- [Tavily Search API](https://docs.tavily.com/)
- [Groq](https://groq.com/) (uses `Mistral`)
- [BeautifulSoup](https://pypi.org/project/beautifulsoup4/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- Python `asyncio`, `os`, `requests`

---

## 🔧 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Aashay98/LLMStockPulse-main.git
cd LLMStockPulse-main
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
```
Windows:
```bash
venv\Scripts\activate
```
Linux/macOS:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
You’ll be prompted at runtime, or you can set them manually:
```bash
export GROQ_API_KEY="your_groq_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export NEWS_API_KEY="your_newsapi_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```
Or use a `.env` file with `python-dotenv`.

---

## ▶️ How to Run

```bash
python -m streamlit run Stock.py
```

This will launch a browser window with a chatbot interface for asking financial questions.

---

## 💬 Example Questions

- `"Analyze Tesla stock and provide a recommendation."`
- `"What is the sentiment around Nvidia this week?"`
- `"Give me the latest financials of Apple."`
- `"Should I invest in Google based on current metrics?"`
- `"What's the market outlook for the semiconductor sector?"`

---

## ⚠️ Notes
- Make sure your API keys are active and within rate limits.
- Tavily and Alpha Vantage have limited free-tier usage.
- This app uses **Groq's Mistral model** via LangChain for better performance and speed.

---

## 📄 License

MIT License
