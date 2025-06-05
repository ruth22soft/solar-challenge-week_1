import streamlit as st
import pandas as pd
import sys, os
# Add the 'scripts' directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.join('..', 'scripts')))

from stock_analysis import load_data, plot_stock_data, plot_rsi, plot_macd
from sentiment_analysis import SentimentAnalyzer as sa  # Import the new functions
# Streamlit UI
def main():
    st.title('Stock Data and Sentiment Analysis')


    # Load data
    df = load_data('../Data/stock_data.csv')
    daily_sentiment = pd.read_csv('../Data/stock_data.csv')
    stocks = df['stock'].unique()

    selected_stock = st.sidebar.selectbox('Select Stock', stocks)
    indicator = st.sidebar.selectbox('Select Indicator', ['Moving Averages', 'RSI', 'MACD', 'Daily Sentiment'])

    if indicator == 'Moving Averages':
        fig = plot_stock_data(selected_stock, df)
    elif indicator == 'RSI':
        fig = plot_rsi(selected_stock, df)
    elif indicator == 'MACD':
        fig = plot_macd(selected_stock, df)
    elif indicator == 'Daily Sentiment':
        fig = sa.plot_sentiment(daily_sentiment, selected_stock)
       
    st.pyplot(fig)
   

if __name__ == "__main__":
    main()