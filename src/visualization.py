import matplotlib.pyplot as plt
import seaborn as sns
import os
from .config import PLOTS_DIR

class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")

    def save_plot(self, fig, filename):
        path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved plot: {path}")

    def plot_medal_trends(self, df, top_n=5):
        top_teams = df.groupby('Team')['Total_Medals'].sum().nlargest(top_n).index
        plt.figure(figsize=(12, 6))
        
        for team in top_teams:
            team_data = df[df['Team'] == team]
            plt.plot(team_data['Year'], team_data['Total_Medals'], marker='o', label=team)
            
        plt.title(f'Medal Trends for Top {top_n} Countries')
        plt.xlabel('Year')
        plt.ylabel('Total Medals')
        plt.legend()
        self.save_plot(plt.gcf(), 'top_countries_medal_trend.png')

    def plot_actual_vs_predicted(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Medals')
        plt.ylabel('Predicted Medals')
        plt.title('Actual vs Predicted Medal Counts')
        self.save_plot(plt.gcf(), 'prediction_scatter.png')
