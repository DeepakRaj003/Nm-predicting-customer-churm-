import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8],
        'MonthlyCharges': [20, 65, 40, 75, 30, 80, 55, 90],
        'Tenure': [12, 5, 24, 3, 18, 7, 30, 1],
        'Churn': [0, 1, 0, 1, 0, 1, 0, 1]}  # 0: No Churn, 1: Churn
df = pd.DataFrame(data)
X = df[['MonthlyCharges', 'Tenure']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
new_customers = pd.DataFrame({'MonthlyCharges': [45, 85], 'Tenure': [10, 2]})
predictions = model.predict(new_customers)
print("\nChurn Predictions for New Customers:") 
for i, pred in enumerate(predictions):
    print(f"Customer {i+1}: {'Churn' if pred == 1 else 'No Churn'}")
