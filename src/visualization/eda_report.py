import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Data
# =========================
df = pd.read_excel("data/raw/Telco_customer_churn.xlsx")

print("\n========== Dataset Overview ==========")
print(df.shape)
print(df.head())

# =========================
# Clean Target Column
# =========================
df["Churn Value"] = df["Churn Value"].astype(int)

# =========================
# 1. CHURN DISTRIBUTION
# =========================
churn_counts = df["Churn Label"].value_counts()

plt.figure()
churn_counts.plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.show()

print("\nInsight: Most customers are non-churned, dataset is imbalanced.")

# =========================
# 2. CHURN BY CONTRACT TYPE
# =========================
contract_churn = df.groupby("Contract")["Churn Value"].mean()

plt.figure()
contract_churn.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")
plt.show()

print("\nInsight: Month-to-month customers are more likely to churn.")

# =========================
# 3. MONTHLY CHARGES vs CHURN
# =========================
plt.figure()
sns.boxplot(x="Churn Label", y="Monthly Charges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

print("\nInsight: Customers with higher charges tend to churn more.")

# =========================
# 4. TENURE vs CHURN
# =========================
plt.figure()
sns.boxplot(x="Churn Label", y="Tenure Months", data=df)
plt.title("Tenure vs Churn")
plt.show()

print("\nInsight: New customers are more likely to churn.")