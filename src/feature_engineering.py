import pandas as pd

class FeatureEngineer:
    def add_features(self, agg_df):
        print("Adding engineered features...")
        agg_df['Medals_Per_Athlete'] = agg_df['Total_Medals'] / agg_df['Athlete_Count']
        
        agg_df = agg_df.sort_values(by=['Team', 'Year'])
        agg_df['Prev_Medals'] = agg_df.groupby('Team')['Total_Medals'].shift(1).fillna(0)
        
        return agg_df
