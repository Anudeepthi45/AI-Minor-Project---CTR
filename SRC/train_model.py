from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess('data/advertising.csv')

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")
