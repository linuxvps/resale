from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# بارگیری داده‌ها
data = load_iris()
X = data.data
y = data.target

# تقسیم داده به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تعریف مدل‌های پایه مختلف
base_models = [
    ('DecisionTree', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('SVM', SVC(probability=True, random_state=42)),
    ('LogisticRegression', LogisticRegression(random_state=42))
]

# ایجاد مدل Voting برای ترکیب نتایج
voting_model = VotingClassifier(
    estimators=base_models,
    voting='hard'  # رأی اکثریت
)

# ایجاد مدل Bagging با استفاده از VotingClassifier به عنوان مدل پایه
bagging_model = BaggingClassifier(
    voting_model,        # مدل پایه (Voting)
    n_estimators=10,     # تعداد بگ‌ها
    max_samples=0.8,     # 80% داده برای هر بگ
    bootstrap=True,      # نمونه‌گیری با جایگزینی
    random_state=42
)

# آموزش مدل
bagging_model.fit(X_train, y_train)

# پیش‌بینی و ارزیابی
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# گزارش طبقه‌بندی
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))