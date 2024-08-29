import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
from datetime import date

start="2015-01-01"
today=date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction App")
stocks=("AAPL","GOOG","MSFT","GME")
selected_stock=st.selectbox("Select dataset for prediction",stocks)
n_years = st.slider("Years of Prediction",1,4)
period=n_years*365
@st.cache
def load_data(ticker):
  data=yf.download(ticker,start,today)
  data.reset_index(inplace=True)
  return data
data_load_state=st.text("Load data...")
data=load_data(selected_stock)
data_load_state.text("Loading data...Done!")
st.subheader('Raw data')
st.write(data.tail())
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

#Forecasting
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast = m.predict((future))
st.subheader('Forecast data')
st.write(forecast.tail())
st.write('Forecast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2=m.plot_components(forecast)
st.write(fig2)