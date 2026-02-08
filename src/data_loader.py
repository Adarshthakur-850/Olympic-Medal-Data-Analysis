import pandas as pd
from .config import DATA_URL

class DataLoader:
    def load_data(self):
        print("Loading data from URL...")
        df = pd.read_csv(DATA_URL)
        return df

    def clean_data(self, df):
        print("Cleaning data...")
        df = df.drop_duplicates()
        
        for col in ['Age', 'Height', 'Weight']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        df['Medal'] = df['Medal'].fillna('No Medal')
        
        return df
