# Olympic Medal Data Analysis


## Project Structure
```
Olympic Medal Data Analysis/
├── models/             # Saved ML models
├── plots/              # Visualization of trends
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py # Aggregation by Country/Year
│   ├── feature_engineering.py # Medals per Athlete, Lag features
│   ├── visualization.py
│   ├── models.py
│   ├── train.py
├── main.py
└── requirements.txt
```

## Features
- **Data Engineering**: Aggregates raw athlete-level data into country-year level summaries.
- **Insights**: Visualization of top performing countries over time.
- **Machine Learning**: Predicts a country's total medal count based on athlete participation and historical performance.
- **Models**: Linear Regression and Random Forest.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```
