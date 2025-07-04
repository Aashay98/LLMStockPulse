# 📊 Stock Insight Multi-Agent App

An intelligent financial assistant that combines **LLMs**, **LangChain**, **Streamlit**, and powerful real-time APIs to deliver actionable stock insights. This multi-agent app acts like a **personal stock analyst**, capable of understanding financial queries, routing them to specialized agents, and responding with structured, up-to-date insights.

## 🚀 Recent Improvements (v2.0)

### 🐛 Bug Fixes
- **Fixed missing import**: Added `partial` import for async URL processing
- **Enhanced error handling**: Comprehensive exception handling across all modules
- **API validation**: Proper validation of API responses and error messages
- **Memory management**: Fixed memory initialization and context saving
- **Stock symbol validation**: Added proper validation for stock ticker symbols
- **Rate limiting**: Implemented retry logic with exponential backoff

### ⚡ Performance Enhancements
- **Caching**: Added `@st.cache_resource` for expensive operations
- **Async processing**: Improved concurrent URL processing
- **Token management**: Better text trimming and token counting
- **Response formatting**: Enhanced data presentation and readability

### 🛡️ Security & Reliability
- **Input validation**: Comprehensive validation for all user inputs
- **Error boundaries**: Graceful error handling with user-friendly messages
- **API key validation**: Startup validation of required environment variables
- **Safe type conversion**: Robust handling of API data types

### 🎨 UI/UX Improvements
- **Enhanced interface**: Better visual design with custom CSS
- **Progress indicators**: Loading spinners for better user feedback
- **Quick actions**: Pre-defined query buttons for common requests
- **Session statistics**: Real-time metrics and error tracking
- **Improved HITL**: Better human-in-the-loop review interface

## 🧠 Architecture Overview

### Multi-Agent System
- **Stock Data Agent**: Real-time prices, fundamentals, technical indicators
- **Sentiment Agent**: News analysis and market sentiment
- **Social Sentiment Agent**: Reddit/Twitter discussions analysis
- **Insights Agent**: Web research and comprehensive analysis
- **General Agent**: Handles diverse financial queries
- **Coordinator Agent**: Manages multi-agent interactions

### Key Features
- 🔍 **Real-time stock data** via Alpha Vantage API
- 📰 **News sentiment analysis** from multiple sources
- 📣 **Social media sentiment** from Reddit and Twitter
- 🧠 **AI-powered insights** using advanced LLMs
- 🤖 **Human-in-the-loop** review and editing
- 📊 **Technical analysis** with RSI, MACD indicators
- 💡 **Investment recommendations** based on multiple factors

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom CSS
- **LLM**: Groq (Llama 3.3 70B)
- **Framework**: LangChain for agent orchestration
- **APIs**: Alpha Vantage, NewsAPI, Tavily Search, Twitter/X
- **ML**: Sentence Transformers, FAISS for similarity search
- **Data**: BeautifulSoup for web scraping
- **Storage**: JSON-based persistent conversation history

## 🔧 Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-repo/stock-insight-app.git
cd stock-insight-app
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file with your API keys:
```env
GROQ_API_KEY=your_groq_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_newsapi_key
TAVILY_API_KEY=your_tavily_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

### 4. Run Application
```bash
streamlit run app.py
```

## 📋 API Requirements

### Required API Keys
1. **Groq API**: For LLM processing - [Get API Key](https://console.groq.com/)
2. **Alpha Vantage**: For stock data - [Get API Key](https://www.alphavantage.co/support/#api-key)
3. **NewsAPI**: For news data - [Get API Key](https://newsapi.org/register)
4. **Tavily**: For web search - [Get API Key](https://tavily.com/)
5. **Twitter/X**: For social sentiment - requires Bearer Token

Reddit data is fetched via the public Pushshift API, so no key is needed.

### API Limits
- Alpha Vantage: 25 requests/day (free tier)
- NewsAPI: 1000 requests/month (free tier)
- Tavily: 1000 requests/month (free tier)
- Groq: Generous free tier with high rate limits
- Twitter/X: subject to standard API usage limits

## 💬 Example Queries

### Stock Analysis
- "Analyze AAPL stock performance and provide investment recommendation"
- "Compare TSLA vs NVDA fundamentals"
- "What's the current P/E ratio and RSI for Microsoft?"

### Market Sentiment
- "What's the market sentiment around Tesla this week?"
- "Analyze recent news impact on tech stocks"
- "How is the market reacting to recent Fed announcements?"

### General Financial
- "Best dividend stocks for 2024"
- "Explain the current market volatility"
- "Should I invest in AI stocks now?"

## 🔍 Features Deep Dive

### Human-in-the-Loop (HITL)
- Review AI responses before they're added to conversation
- Edit and improve responses for accuracy
- Track editing history and patterns
- Conversation history persisted on disk (cleared via sidebar button)
- Toggle between automatic and review modes

### Multi-Agent Coordination
- Intelligent query classification
- Parallel agent execution for comprehensive analysis
- Response synthesis from multiple sources
- Fallback mechanisms for reliability

### Advanced Analytics
- Technical indicator analysis (RSI, MACD)
- Fundamental analysis (P/E, Market Cap, EPS)
- Sentiment scoring from news sources
- Investment recommendation engine

## 🚨 Error Handling

The application includes comprehensive error handling:
- API rate limit management
- Network timeout handling
- Invalid input validation
- Graceful degradation when services are unavailable
- User-friendly error messages

## 📊 Monitoring & Logging

- Session statistics tracking
- Error count monitoring
- Query classification metrics
- HITL edit logging
- Performance monitoring

## 🔒 Security Considerations

- API key validation at startup
- Input sanitization and validation
- Safe type conversions
- Error message sanitization
- No sensitive data logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper testing
4. Update documentation
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check the error logs in the sidebar
2. Verify all API keys are valid and have remaining quota
3. Review the troubleshooting section
4. Open an issue on GitHub

---

**Disclaimer**: This application is for educational and informational purposes only. It should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.