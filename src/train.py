import pickle
import os
from sklearn.model_selection import train_test_split
from .config import MODELS_DIR, TEST_SIZE, RANDOM_STATE

class Trainer:
    def split_data(self, df, target='Total_Medals'):
        X = df.drop(columns=[target, 'Team']) 
        y = df[target]
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def save_model(self, model, filename):
        path = os.path.join(MODELS_DIR, filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {path}")
