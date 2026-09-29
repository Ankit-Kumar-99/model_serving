import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

# Define the custom MLflow wrapper class
class ARIMAForecastModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        # Load the saved model pickle file from the MLflow artifacts
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        import pandas as pd
        # Extract the 'steps' integer from the incoming DataFrame
        steps = int(model_input["steps"].iloc[0])
        # Generate the forecast
        forecast = self.model.forecast(steps)
        # Return as a clean DataFrame
        return pd.DataFrame({"forecast": forecast.values})

# Format dates
df = pd.read_csv("exchange_rate.csv")
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
df.set_index("date", inplace=True)
df = df.dropna().sort_index()

# Remove outliers
lower_limit = df["Ex_rate"].quantile(0.01)
upper_limit = df["Ex_rate"].quantile(0.99)
df["Ex_rate"] = np.clip(df["Ex_rate"], lower_limit, upper_limit)

# Train-test split
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

mlflow.set_experiment("TSA_Exchange_Rate")

with mlflow.start_run():
    # Train model
    model = ARIMA(train["Ex_rate"], order=(1, 1, 1)).fit()
    predictions = model.forecast(len(test))

    # Calculate errors
    mae = mean_absolute_error(test["Ex_rate"], predictions)
    rmse = np.sqrt(mean_squared_error(test["Ex_rate"], predictions))
    mape = mean_absolute_percentage_error(test["Ex_rate"], predictions)

    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.6f}")

    # Log to MLflow
    mlflow.log_param("order", "(1, 1, 1)")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)

    final_model = ARIMA(df["Ex_rate"], order=(1, 1, 1)).fit()

    # Save model locally so the custom PyFunc can bundle it
    with open("model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    # Log model to MLflow using the custom PyFunc
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ARIMAForecastModel(),
        artifacts={"model": "model.pkl"}
    )

    # Forecast future
    future_predictions = final_model.forecast(30)
    print("\nNext 30 Days Forecast:")
    print(future_predictions)
