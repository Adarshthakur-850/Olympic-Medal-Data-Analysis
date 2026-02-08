import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self):
        self.le_dict = {}

    def encode_categorical(self, df, columns=['Sex', 'Team', 'Sport', 'Event']):
        print("Encoding categorical features...")
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.le_dict[col] = le
        return df

    def aggregate_by_country_year(self, df):
        print("Aggregating data by Country (Team) and Year...")
        medals_df = df[df['Medal'] != 'No Medal']
        
        medal_counts = medals_df.groupby(['Year', 'Team'])['Medal'].count().reset_index()
        medal_counts.rename(columns={'Medal': 'Total_Medals'}, inplace=True)
        
        athlete_counts = df.groupby(['Year', 'Team'])['ID'].nunique().reset_index()
        athlete_counts.rename(columns={'ID': 'Athlete_Count'}, inplace=True)
        
        agg_df = pd.merge(athlete_counts, medal_counts, on=['Year', 'Team'], how='left')
        agg_df['Total_Medals'] = agg_df['Total_Medals'].fillna(0)
        
        return agg_df
