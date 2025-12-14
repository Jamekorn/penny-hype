import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1E3A8A;
    }
    .buy-signal {
        color: #10B981;
        font-weight: bold;
    }
    .sell-signal {
        color: #EF4444;
        font-weight: bold;
    }
    .hold-signal {
        color: #F59E0B;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Hybrid Stock Prediction Framework</h1>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Ticker input
    ticker = st.text_input("Enter Stock Ticker", value="GME").upper()
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Prediction Timeframe",
        ["Short-term (1 day)", "Long-term (1 week)"],
        index=0
    )
    
    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["Hybrid (Hype + Reality)", "Hype Only", "Reality Only"],
        index=0
    )
    
    # Feature toggles
    st.subheader("Feature Toggles")
    include_social = st.checkbox("Include Social Features", value=True)
    include_fundamentals = st.checkbox("Include Fundamental Features", value=True)
    include_bot_detection = st.checkbox("Include Bot Detection", value=True)
    
    # Refresh button
    refresh = st.button("🔍 Analyze Stock", type="primary")

# API URL (update this based on your deployment)
API_URL = "http://localhost:8000"

def get_prediction(ticker: str, timeframe: str) -> Dict:
    """Get prediction from API"""
    timeframe_param = "short" if "Short-term" in timeframe else "long"
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "ticker": ticker,
                "timeframe": timeframe_param,
                "include_social": include_social,
                "include_fundamentals": include_fundamentals
            }
        )
        return response.json()
    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None

def get_features(ticker: str) -> Dict:
    """Get features from API"""
    try:
        response = requests.get(f"{API_URL}/features/{ticker}")
        return response.json()
    except Exception as e:
        st.error(f"Error getting features: {e}")
        return None

def display_prediction_card(prediction: Dict):
    """Display prediction in a card"""
    pred = prediction["prediction"]
    confidence = prediction["confidence"]
    
    if pred == "BUY":
        signal_class = "buy-signal"
        emoji = "🟢"
    elif pred == "SELL":
        signal_class = "sell-signal"
        emoji = "🔴"
    else:
        signal_class = "hold-signal"
        emoji = "🟡"
    
    st.markdown(f"""
    <div class="prediction-card">
        <h2>{emoji} Prediction: <span class="{signal_class}">{pred}</span></h2>
        <p><strong>Confidence:</strong> {confidence:.2%}</p>
        <p><strong>Ticker:</strong> {prediction['ticker']}</p>
        <p><strong>Model:</strong> {prediction['model_type']}</p>
        <p><strong>Time:</strong> {prediction['timestamp']}</p>
    </div>
    """, unsafe_allow_html=True)

def plot_feature_importance(features: Dict):
    """Plot feature importance"""
    # Sort features by absolute value
    sorted_features = sorted(
        features.items(),
        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        reverse=True
    )[:15]  # Top 15 features
    
    feature_names = [f[0] for f in sorted_features]
    feature_values = [f[1] if isinstance(f[1], (int, float)) else 0 for f in sorted_features]
    
    colors = ['green' if v > 0 else 'red' for v in feature_values]
    
    fig = go.Figure(
        data=[go.Bar(
            x=feature_values,
            y=feature_names,
            orientation='h',
            marker_color=colors
        )]
    )
    
    fig.update_layout(
        title="Top Feature Contributions",
        xaxis_title="Feature Value",
        yaxis_title="Feature Name",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_timeline():
    """Plot sentiment timeline (mock data)"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    sentiment = np.random.randn(30).cumsum()
    volume = np.random.randint(50, 200, size=30)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Sentiment Over Time", "Social Volume"),
        vertical_spacing=0.1
    )
    
    # Sentiment line
    fig.add_trace(
        go.Scatter(x=dates, y=sentiment, mode='lines', name='Sentiment'),
        row=1, col=1
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(x=dates, y=volume, name='Volume'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_yaxes(title_text="Post Count", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_feature_summary(features: Dict):
    """Display feature summary"""
    # Categorize features
    hype_features = {k: v for k, v in features.items() if 'hype' in k or 'sentiment' in k}
    reality_features = {k: v for k, v in features.items() if 'reality' in k or 'risk' in k or 'dilution' in k}
    bot_features = {k: v for k, v in features.items() if 'bot' in k}
    
    # Create columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Hype Score",
            f"{np.mean(list(hype_features.values())):.2f}" if hype_features else "N/A",
            delta="+0.15" if np.random.rand() > 0.5 else "-0.15"
        )
    
    with col2:
        st.metric(
            "Reality Score",
            f"{np.mean(list(reality_features.values())):.2f}" if reality_features else "N/A",
            delta="+0.08" if np.random.rand() > 0.5 else "-0.08"
        )
    
    with col3:
        st.metric(
            "Bot Probability",
            f"{np.mean(list(bot_features.values())):.2f}" if bot_features else "N/A",
            delta="+0.05" if np.random.rand() > 0.5 else "-0.05"
        )
    
    # Detailed metrics
    with st.expander("Detailed Feature Analysis"):
        tab1, tab2, tab3 = st.tabs(["Hype Features", "Reality Features", "Bot Detection"])
        
        with tab1:
            if hype_features:
                hype_df = pd.DataFrame(
                    list(hype_features.items()),
                    columns=['Feature', 'Value']
                )
                st.dataframe(hype_df, use_container_width=True)
            else:
                st.info("No hype features available")
        
        with tab2:
            if reality_features:
                reality_df = pd.DataFrame(
                    list(reality_features.items()),
                    columns=['Feature', 'Value']
                )
                st.dataframe(reality_df, use_container_width=True)
            else:
                st.info("No reality features available")
        
        with tab3:
            if bot_features:
                bot_df = pd.DataFrame(
                    list(bot_features.items()),
                    columns=['Feature', 'Value']
                )
                st.dataframe(bot_df, use_container_width=True)
            else:
                st.info("No bot detection features available")

# Main content
if refresh or st.session_state.get('auto_refresh', False):
    with st.spinner(f"Analyzing {ticker}..."):
        # Get prediction
        prediction = get_prediction(ticker, timeframe)
        
        if prediction:
            # Display prediction card
            display_prediction_card(prediction)
            
            # Get features for detailed analysis
            features_data = get_features(ticker)
            
            if features_data and 'features' in features_data:
                features = features_data['features']
                
                # Feature summary
                st.subheader("Feature Summary")
                display_feature_summary(features)
                
                # Feature importance plot
                st.subheader("Feature Importance")
                plot_feature_importance(features)
                
                # Sentiment timeline
                st.subheader("Sentiment Timeline")
                plot_sentiment_timeline()
            
            # Recommendation section
            st.subheader("📋 Recommendation Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **Key Strengths:**
                - Strong social sentiment momentum
                - Low bot activity detected
                - Positive catalyst identified
                """)
            
            with col2:
                st.warning("""
                **Risks to Consider:**
                - High volatility expected
                - Dilution risk present
                - Limited cash runway
                """)
            
            # Action buttons
            st.markdown("---")
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("📊 View Detailed Report", type="secondary"):
                    st.session_state.show_report = True
            
            with action_col2:
                if st.button("🔔 Set Alert", type="secondary"):
                    st.info("Alert feature coming soon!")
            
            with action_col3:
                if st.button("🔄 Compare with Peers", type="secondary"):
                    st.info("Comparison feature coming soon!")
else:
    # Welcome screen
    st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze Stock' to get started!")
    
    # Quick analysis section
    st.subheader("Quick Analysis")
    
    quick_tickers = ["GME", "AMC", "TSLA", "AAPL", "NVDA"]
    cols = st.columns(len(quick_tickers))
    
    for idx, ticker in enumerate(quick_tickers):
        with cols[idx]:
            if st.button(f"📈 {ticker}", key=f"quick_{ticker}"):
                st.session_state.quick_ticker = ticker
                st.session_state.auto_refresh = True
                st.rerun()
    
    # Recent predictions
    st.subheader("Recent Predictions")
    
    recent_data = [
        {"ticker": "GME", "prediction": "BUY", "confidence": 0.78, "time": "2 hours ago"},
        {"ticker": "AMC", "prediction": "SELL", "confidence": 0.65, "time": "4 hours ago"},
        {"ticker": "TSLA", "prediction": "HOLD", "confidence": 0.52, "time": "6 hours ago"},
        {"ticker": "NVDA", "prediction": "BUY", "confidence": 0.81, "time": "1 day ago"},
    ]
    
    for pred in recent_data:
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            with col1:
                st.write(f"**{pred['ticker']}**")
            with col2:
                color = "green" if pred['prediction'] == 'BUY' else "red" if pred['prediction'] == 'SELL' else "orange"
                st.markdown(f"<span style='color:{color}'>{pred['prediction']}</span>", unsafe_allow_html=True)
            with col3:
                st.write(f"{pred['confidence']:.0%}")
            with col4:
                st.write(pred['time'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Hybrid Stock Prediction Framework • AY 25/26 • Jamekorn Likitwattananurak, Glen Tan Lusheng, Linh Bui</p>
    <p>Data updates every 15 minutes • Predictions are for educational purposes only</p>
</div>
""", unsafe_allow_html=True)