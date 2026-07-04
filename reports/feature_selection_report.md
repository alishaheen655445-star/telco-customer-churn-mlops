# Feature Selection Report

## Removed Features

| Feature | Reason |
|----------|--------|
| CustomerID | Identifier |
| Count | Constant value |
| Churn Label | Duplicate target |
| Churn Score | Data leakage |
| Churn Reason | Available after churn |
| Lat Long | Duplicate coordinates |
| Country | Constant value |
| State | Constant value |

---

## Features Under Review

| Feature | Reason |
|----------|--------|
| City | High cardinality |
| Zip Code | High cardinality |
| CLTV | Possible data leakage |