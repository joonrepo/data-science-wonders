# --- Imports (only once, at the top) ---
import pandas as pd
from sklearn.metrics import precision_score

# --- Create Target Column ---
def create_target_column(stocks_list):
    """
    Processes a list of stock DataFrames and creates a 'Target' column.
    'Target' = 1 if tomorrow's close > today's close, else 0.
    """
    for idx, df_stock in enumerate(stocks_list):
        df_stock['Tomorrow'] = df_stock['Close'].shift(-1)
        df_stock['Target'] = (df_stock['Tomorrow'] > df_stock['Close']).astype(int)
        df_stock.dropna(subset=['Target'], inplace=True)
        df_stock = df_stock.loc['1990-01-01':].copy()
        stocks_list[idx] = df_stock

    print("Target columns created for all stocks")
    return stocks_list

# --- Predict Function ---
def predict(train, test, predictors, model):
    """
    Trains the model and predicts on the test set.
    """
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined
print('Predict function created.')

# --- Backtest Function ---
def backtest(data, model, predictors, start=3750, step=100):
    """
    Simulates backtesting by training and predicting in rolling windows.
    """
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
print('Backtest function created.')

# --- Run Model for All Stocks ---
def run_model_for_all_stocks(stocks_list, model, predictors):
    """
    Runs the model on all stocks and returns the top 5 by precision score.
    """
    stock_accuracies = []

    for idx, df_stock in enumerate(stocks_list):
        print(f"Running model for stock {idx+1}")
        df_stock_copy = df_stock.copy()
        predictions = backtest(df_stock_copy, model, predictors)
        precision = precision_score(predictions['Target'], predictions['Predictions'], average='binary', pos_label=1)
        stock_accuracies.append((f"Stock_{idx+1}", precision))

    top_5_stocks = sorted(stock_accuracies, key=lambda x: x[1], reverse=True)[:5]
    return top_5_stocks

# --- Add More Predictors Function ---
def add_more_predictors(df):
    """
    Adds additional predictors (Prev_Close_Return, Intraday_Volatility, Volume_Spike_Ratio_5).
    """
    df = df.copy()
    additional_predictors = []

    df["Prev_Close_Return"] = df["Close"].pct_change()
    additional_predictors.append("Prev_Close_Return")

    df["Intraday_Volatility"] = (df["High"] - df["Low"]) / df["Open"]
    additional_predictors.append("Intraday_Volatility")

    df["Volume_Spike_Ratio_5"] = df["Volume"] / df["Volume"].rolling(window=5).mean()
    additional_predictors.append("Volume_Spike_Ratio_5")

    df.dropna(inplace=True)
    return df, additional_predictors
print("Function to add more predictors has been created.")


## -- This is to test the git --
def this_is_test():
    greeting = 'hello world'
    return greeting

## -- This is second test --
def this_is_second_test():
    greeting = 'hello world two'
    return greeting