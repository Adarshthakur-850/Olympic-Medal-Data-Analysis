from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.visualization import Visualizer
from src.models import ModelFactory
from src.train import Trainer
import pandas as pd

def main():
    print("=== Olympic Medal Data Analysis ===")
    
    loader = DataLoader()
    raw_df = loader.load_data()
    raw_df = loader.clean_data(raw_df)
    
    preprocessor = Preprocessor()
    agg_df = preprocessor.aggregate_by_country_year(raw_df)
    
    fe = FeatureEngineer()
    agg_df = fe.add_features(agg_df)
    print(f"Aggregated Data Shape: {agg_df.shape}")
    print(agg_df.head())
    
    viz = Visualizer()
    viz.plot_medal_trends(agg_df, top_n=5)
    
    trainer = Trainer()
    
    agg_df = agg_df.dropna()
    
    X_train, X_test, y_train, y_test = trainer.split_data(agg_df, target='Total_Medals')
    
    models = ['linear_regression', 'random_forest']
    best_r2_score = -float('inf')
    best_model_obj = None
    
    for m_name in models:
        print(f"\n--- Training {m_name} ---")
        model = ModelFactory.get_model(m_name)
        model = trainer.train_model(model, X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae, rmse, r2 = ModelFactory.evaluate(y_test, y_pred)
        
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        viz.plot_actual_vs_predicted(y_test, y_pred)
        
        if r2 > best_r2_score:
            best_r2_score = r2
            best_model_obj = model
            
    if best_model_obj:
        trainer.save_model(best_model_obj, 'best_medal_predictor.pkl')

    print("\n=== Analysis Completed ===")

if __name__ == "__main__":
    main()
