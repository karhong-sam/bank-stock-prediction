import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
import pickle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def fetch_and_process_data(stock_ticker, start_date, end_date):
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    st.info("Data fetching in progress...")
    st.write(f"Fetching data for {stock_ticker}...")
    data = yf.download(stock_ticker, start=start_date, end=end_date)

    if not data.empty:
        # Normalize column names to ensure consistency
        data.columns = [col.split(" ")[-1] if " " in col else col for col in data.columns]

        st.write("### Data Preview")
        st.dataframe(data.head())
        st.write(f"Data fetched with {len(data)} rows and {len(data.columns)} columns.")

        # Feature Engineering
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA60'] = data['Close'].rolling(window=60).mean()
        data = data.dropna()

        st.write("Data fetching complete!")
        st.session_state['data'] = data
    else:
        st.error("No data found for the selected ticker and dates.")

def plot_moving_averages(data, filtered_data):
    if filtered_data.empty:
        return None  # Return None if there's no data to plot

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], mode='lines', name='Close Price'))
    if 'MA20' in filtered_data.columns:
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['MA20'], mode='lines', name='MA20'))
    if 'MA60' in filtered_data.columns:
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['MA60'], mode='lines', name='MA60'))
    fig.update_layout(
        title="Close Price with MA20 and MA60",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Indicators",
        hovermode="x unified"
    )
    return fig

def plot_macd_histogram(filtered_data):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=filtered_data.index, y=filtered_data['MACD_Histogram'], name='MACD Histogram'))
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Signal_Line'], mode='lines', name='Signal Line'))
    fig.update_layout(
        title="MACD Histogram with Signal Line",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Indicators",
        hovermode="x unified"
    )
    return fig

def view_eda():
    if 'data' in st.session_state:
        data = st.session_state['data']

        # Check if required columns exist
        required_columns = ['EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'MACD_Histogram', 'MA20', 'MA60']
        if not all(col in data.columns for col in required_columns):
            st.error("Required engineered features are missing. Please ensure they are calculated.")
            return

        st.write("### Engineered Features")
        st.dataframe(data[required_columns].head())

        # Filter data by date
        st.subheader("Filter Data by Date")
        min_date = data.index.min()
        max_date = data.index.max()
        if not isinstance(min_date, pd.Timestamp) or not isinstance(max_date, pd.Timestamp):
            st.error("The data index is not in datetime format. Please ensure it is converted to datetime.")
            return

        date_range = st.date_input("Select Date Range", [min_date.to_pydatetime(), max_date.to_pydatetime()])
        filtered_data = data.loc[pd.to_datetime(date_range[0]):pd.to_datetime(date_range[1])]

        if filtered_data.empty:
            st.warning("No data available for the selected date range.")
            return

        # Plot moving averages
        st.subheader("Plot Moving Averages")
        ma_fig = plot_moving_averages(data, filtered_data)
        if ma_fig:
            st.plotly_chart(ma_fig)
        else:
            st.warning("Moving averages could not be plotted due to missing data.")

        # Plot MACD Histogram
        st.subheader("Plot MACD Histogram")
        macd_fig = plot_macd_histogram(filtered_data)
        if macd_fig:
            st.plotly_chart(macd_fig)
        else:
            st.warning("MACD Histogram could not be plotted due to missing data.")
    else:
        st.error("Please fetch data first.")

def plot_loss(train_losses, val_losses, num_epochs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=train_losses, mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=val_losses, mode='lines', name='Validation Loss'))
    fig.update_layout(
        title="Training vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Loss Type",
        hovermode="x unified"
    )
    return fig

def prepare_data(data):
    features = ['Open', 'High', 'Low', 'MACD', 'EMA_12', 'Signal_Line', 'MACD_Histogram']
    target = 'Close'
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data[features + [target]])
    df_normalized = pd.DataFrame(data_normalized, columns=features + [target])

    train_size = int(len(df_normalized) * 0.7)
    validation_size = int(len(df_normalized) * 0.15)
    train = df_normalized[:train_size]
    validate = df_normalized[train_size:train_size + validation_size]
    test = df_normalized[train_size + validation_size:]
    return features, target, train, validate, test, scaler

def train_model(train, validate, features, target):
    input_size = len(features)
    hidden_size = 128
    output_size = 1
    num_layers = 2
    learning_rate = 1e-3
    num_epochs = 100

    X_train = torch.tensor(train[features].values).float().unsqueeze(1).to(device)
    y_train = torch.tensor(train[target].values).float().unsqueeze(1).to(device)
    X_val = torch.tensor(validate[features].values).float().unsqueeze(1).to(device)
    y_val = torch.tensor(validate[target].values).float().unsqueeze(1).to(device)

    model = LSTMRNN(input_size, hidden_size, output_size, num_layers).to(device)

    # Store the model architecture in session state for later use
    st.session_state['model_architecture'] = LSTMRNN(input_size, hidden_size, output_size, num_layers)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_output = model(X_train)
        train_loss = criterion(train_output, y_train)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()

        if (epoch + 1) % 10 == 0:
            st.write(f"Epoch [{epoch+1}/100] - Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    return model, train_losses, val_losses, best_val_loss

def evaluate_model(model, train_losses, val_losses, best_val_loss, scaler, test, features, target):
    st.write("### Training Complete")
    st.write(f"Best Validation Loss: {best_val_loss:.4f}")
    st.write(f"Highest Training Loss: {max(train_losses):.4f}, Lowest Training Loss: {min(train_losses):.4f}")
    st.write(f"Highest Validation Loss: {max(val_losses):.4f}, Lowest Validation Loss: {min(val_losses):.4f}")

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=list(range(1, len(train_losses) + 1)), y=train_losses, mode='lines', name='Train Loss'))
    loss_fig.add_trace(go.Scatter(x=list(range(1, len(val_losses) + 1)), y=val_losses, mode='lines', name='Validation Loss'))
    loss_fig.update_layout(
        title="Training vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Loss Type",
        hovermode="x unified"
    )
    st.plotly_chart(loss_fig)

    save_model_buffer = BytesIO()
    pickle.dump(model.state_dict(), save_model_buffer)
    save_model_buffer.seek(0)

    st.download_button(
        label="Download Trained Model",
        data=save_model_buffer,
        file_name="best_lstm_rnn_model.pkl",
        mime="application/octet-stream"
    )

def train_and_evaluate():
    data = st.session_state['data']
    features, target, train, validate, test, scaler = prepare_data(data)
    df_normalized = pd.concat([train, validate, test])
    st.session_state['data_normalized'] = df_normalized
    model, train_losses, val_losses, best_val_loss = train_model(train, validate, features, target)
    evaluate_model(model, train_losses, val_losses, best_val_loss, scaler, test, features, target)
    
    # Save the trained model to session state
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['test'] = test
    st.session_state['features'] = features

def test_model_performance():
    # Check if the model exists in session state
    if 'model' not in st.session_state:
        st.error("Model not found. Please train the model first or upload a saved model.")
        return

    model = st.session_state['model']
    scaler = st.session_state['scaler']
    test = st.session_state['test']
    features = st.session_state['features']

    X_test = torch.tensor(test[features].values).float().unsqueeze(1).to(device)
    y_test = test['Close'].values

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze().cpu().numpy()

    y_test_denorm = scaler.inverse_transform(
        np.hstack((np.zeros((len(y_test), len(features))), y_test.reshape(-1, 1)))
    )[:, -1]

    y_pred_denorm = scaler.inverse_transform(
        np.hstack((np.zeros((len(y_pred), len(features))), y_pred.reshape(-1, 1)))
    )[:, -1]

    mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
    mse = mean_squared_error(y_test_denorm, y_pred_denorm)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_denorm - y_pred_denorm) / y_test_denorm.clip(min=1e-8))) * 100
    r2 = r2_score(y_test_denorm, y_pred_denorm)

    st.write("### Test Performance")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAPE: {mape:.2f}%")
    st.write(f"R2 Score: {r2:.4f}")

    st.subheader("Plot Predicted vs Actual Close Prices")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=y_test_denorm, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test.index, y=y_pred_denorm, mode='lines', name='Predicted'))

    fig.update_layout(
        title="Predicted vs Actual Close Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        hovermode="x unified"
    )
    st.plotly_chart(fig)

def predict_future_prices():
    model_file = st.file_uploader("Upload a trained model file (.pkl)", type=["pkl"])

    if model_file is not None:
        try:
            model_architecture = st.session_state.get('model_architecture', None)
            if model_architecture is None:
                st.error("Model architecture is not available. Please train the model first.")
                return
            
            loaded_model = model_architecture.to(device)
            loaded_model.load_state_dict(pickle.load(model_file))
            st.session_state['model'] = loaded_model
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    if 'model' in st.session_state:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        features = st.session_state['features']
        df_normalized = st.session_state['data_normalized']

        predict_days = st.slider("Predict days into the future", min_value=1, max_value=150, value=14)

        if st.button("Predict"):
            last_row = df_normalized.iloc[-1][features].values.reshape(1, -1)
            predictions = []
            for _ in range(predict_days):
                next_pred = model(torch.tensor(last_row).float().unsqueeze(0).to(device)).item()
                predictions.append(next_pred)
                last_row = np.append(last_row[:, 1:], [[next_pred]], axis=1)

            predictions_denorm = scaler.inverse_transform(
                np.hstack((np.zeros((len(predictions), len(features))), np.array(predictions).reshape(-1, 1)))
            )[:, -1]

            st.write("### Predicted Prices")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, predict_days + 1)), y=predictions_denorm, mode='lines', name='Predicted Prices'))

            fig.update_layout(
                title="Future Predicted Prices",
                xaxis_title="Days Ahead",
                yaxis_title="Price",
                legend_title="Legend",
                hovermode="x unified"
            )
            st.plotly_chart(fig)
    else:
        st.error("Please train or upload a model first.")

