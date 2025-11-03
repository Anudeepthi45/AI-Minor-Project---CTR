import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    drop_cols = ['Ad Topic Line', 'City', 'Country', 'Timestamp']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    X = df.drop('Clicked on Ad', axis=1)
    y = df['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    return X_train, X_test, y_train, y_test, scaler
