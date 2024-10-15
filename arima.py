import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore")


def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce', format='mixed')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Adj Close Scaled'] = scaler.fit_transform(df[['Adj Close']])
    return df, scaler


def plot_results(train_dates, train_actual, test_dates, test_actual, test_predicted, conf_int, title, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(train_dates.to_pydatetime(), train_actual, label='Training Data', color='blue')
    plt.plot(test_dates.to_pydatetime(), test_actual, label='Actual', color='blue', linestyle='--')
    plt.plot(test_dates.to_pydatetime(), test_predicted, label='Predicted', color='red', linestyle='--')
    plt.fill_between(test_dates.to_pydatetime(), conf_int[:, 0], conf_int[:, 1], color='k', alpha=0.1, label='Confidence Interval')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def select_model_params(crypto_name):
    best_models = { #tested and 7 is to account for weekly seasonality 
        'ADA': (1, 1, 1, (0, 0, 1, 7)),
        'BNB': (3, 1, 4, (0, 0, 0, 7)),
        'BTC': (0, 1, 0, (0, 0, 0, 7)),
        'ETH': (1, 1, 0, (0, 0, 0, 7)),
        'SOL': (0, 1, 0, (0, 0, 0, 7)),
        'XRP': (1, 1, 0, (1, 0, 1, 7)),
    }
    return best_models.get(crypto_name, (1, 1, 1, (0, 0, 0, 7)))


def process_file(file_path, output_dir, summary):
    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error: Issue with file {file_path}")
        return

    df, scaler = preprocess_data(df)
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]['Adj Close Scaled']
    test_data = df.iloc[split_idx:]['Adj Close Scaled']
    train_dates = df.index[:split_idx]
    test_dates = df.index[split_idx:]

    crypto_name = os.path.basename(file_path).split('-')[0]
    print(f"Processing {crypto_name}...")

    p, d, q, seasonal_order = select_model_params(crypto_name)
    model = ARIMA(train_data, order=(p, d, q), seasonal_order=seasonal_order)
    model_fit = model.fit()

    forecast = model_fit.get_forecast(steps=len(test_data))
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()

    forecast_values = scaler.inverse_transform(forecast_values.values.reshape(-1, 1))
    conf_int = scaler.inverse_transform(conf_int.values)
    test_actual = scaler.inverse_transform(test_data.values.reshape(-1, 1))

    rmse = np.sqrt(root_mean_squared_error(test_actual, forecast_values))
    print(f"RMSE for {crypto_name}: {rmse:.4f}")

    output_plot_path = os.path.join(output_dir, f'{crypto_name}_arima_plot.png')
    plot_results(
        train_dates=train_dates,
        train_actual=scaler.inverse_transform(train_data.values.reshape(-1, 1)).flatten(),
        test_dates=test_dates,
        test_actual=test_actual.flatten(),
        test_predicted=forecast_values.flatten(),
        conf_int=conf_int,
        title=f'{crypto_name} - Actual vs Predicted Adj Close with Confidence Interval',
        output_path=output_plot_path
    )

    prediction_df = pd.DataFrame({'Date': test_dates, 'Actual': test_actual.flatten(), 'Predicted': forecast_values.flatten()})
    output_file = os.path.join(output_dir, f'{crypto_name}_arima_predictions.csv')
    prediction_df.to_csv(output_file, index=False)
    print(f"Finished processing {crypto_name}. Results saved to {output_file}")

    # Add summary data
    summary.append({
        'Crypto': crypto_name,
        'RMSE': rmse,
        'Model': f'ARIMA({p},{d},{q})({seasonal_order})'
    })


def main():
    data_dir = './data'
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith("_historical_data.csv"):
            file_path = os.path.join(data_dir, file_name)
            process_file(file_path, output_dir, summary)

    # Save the summary to a CSV file
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, 'model_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary of results saved to {summary_file}")


if __name__ == "__main__":
    main()
