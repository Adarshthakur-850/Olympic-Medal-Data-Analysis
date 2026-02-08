from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'linear_regression':
            return LinearRegression()
        elif model_name == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def evaluate(y_test, y_pred):
        import numpy as np
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return mae, rmse, r2
