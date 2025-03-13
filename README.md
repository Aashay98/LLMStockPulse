# 📊 Stock Insight Multi-Agent App

This project is an intelligent, multi-agent financial assistant that brings together the power of **LLMs**, **LangChain**, **Streamlit**, and real-time APIs to provide rich, insightful answers to your stock market–related questions.

At its core, this app is designed to **understand your query**, **route it to the most suitable AI agent**, and **respond with well-structured data, news sentiment, and actionable insights**.

Whether you're an investor, a financial researcher, or just curious about market trends, this app acts like a **personal stock analyst**, combining:

- 🔍 **Live data** from financial APIs  
- 🧠 **AI-powered analysis and decision-making**  
- 💬 **Conversational interface for easy interaction**  

The app can:
- Track **real-time stock prices** (intraday & daily)
- Analyze **technical indicators** like RSI
- Retrieve **company financial fundamentals**
- Detect **market sentiment** through recent news
---

## 🚀 Features

- 📈 **Stock Data Fetching**  
  Get up-to-date intraday prices, daily history, technical indicators (like RSI), and financial fundamentals using Alpha Vantage.

- 📰 **Market Sentiment & News**  
  Uses Alpha Vantage and NewsAPI to find the most recent and relevant news, and analyze sentiment for individual tickers or market topics.

- 🔎 **Web & Document Insight Extraction**  
  Search results and content processing are powered by Tavily and BeautifulSoup, allowing you to extract key facts from webpages.

- 🤖 **Multi-Agent Architecture**  
  The app features multiple LangChain agents with different specializations:
  - **Stock Data Agent**: Fetches and interprets financial data
  - **Sentiment Agent**: Analyzes sentiment and media trends
  - **Insights Agent**: Synthesizes deeper insights using search tools
  - **General-Purpose Agent**: Handles any broad or unrelated queries

- 🧠 **Query Classification System**  
  Automatically identifies what type of question you’re asking (stock, sentiment, or general) and assigns it to the appropriate agent.

- 🌐 **Streamlit Web Interface**  
  Chat-based interface makes it easy to ask questions like “What’s the outlook for Apple stock?” and get rich, AI-powered answers instantly.

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Alpha Vantage API](https://www.alphavantage.co/)
- [NewsAPI](https://newsapi.org/)
- [Tavily Search API](https://docs.tavily.com/)
- [Groq (Mistral Model)](https://groq.com/)
- Python, asyncio, BeautifulSoup

---

## 🔑 Setup & Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/stock-insight-agent.git
cd stock-insight-agent
```

2. **Create Vitrual Environment**

```bash
python -m venv venv
```
Activate for Windows:-
```bash
venv\Scripts\activate
```
Activate for Linux/MacOS:-
```bash
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Environment Variables**

You will be prompted for API keys at runtime, but you can also set them manually:

```bash
export GROQ_API_KEY="your_groq_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export NEWS_API_KEY="your_newsapi_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

Or use `.env` and `python-dotenv` if preferred.

---

## ▶️ How to Run

```bash
python -m streamlit run Stock.py
```

This will launch a browser window where you can start chatting with the AI about stocks, news sentiment, and more.

---

## 💡 Example Questions

- `"What is the latest on Tesla stock?"`  
- `"Analyze the sentiment for Apple stock."`  
- `"Should I buy Google shares based on recent data?"`  
- `"What are the top tech stock trends today?"`

---

## 📌 Notes

- Make sure your API keys are active and within rate limits.
- Tavily and Alpha Vantage have limited free-tier usage.
- This app uses **Groq's Mistral model** via LangChain for better performance and speed.

---

## 📄 License

MIT License

---
