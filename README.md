# Waze User Churn Analysis: Predicting App Attrition

I developed a machine learning classification pipeline to predict user churn for the Waze navigation app. By isolating specific behavioral and usage patterns of departing users, this framework provides product and growth teams with early-warning indicators to optimize retention strategies and improve user engagement lifecycles.

###  Model Optimization Tuning & Boundary Trade-offs
I evaluated Random Forest and XGBoost architectures across a stratified 60/20/20 data split derived from 15,000 baseline user rows. 

Because standard classification setups default to a 0.50 probability threshold, they failed to detect true user turnover vectors, leading to low model sensitivity:
* The default Random Forest model hit an 81.8% accuracy but returned a very low Recall score of 12.6%.
* The default XGBoost model delivered an 80.9% accuracy and a 17.3% recall score.

In a consumer application retention project, missing a user who is preparing to uninstall the app (a False Negative) carries a significantly higher business cost than flagging a safe user (a False Positive). To fix this, I ran an automated grid tuning pass to shift the model's internal decision boundary down from 0.50 to **0.194**. This threshold calibration successfully more than doubled our target model sensitivity on the test dataset split:

* **Final Model Precision:** 33.5%
* **Final Model Recall:** 41.4%
* **Final Test Set Accuracy:** 75.0%

**Engineering Conclusion:** By accepting a manageable 5.9% drop in overall classification accuracy (from 80.9% down to 75.0%), we successfully more than doubled our model's sensitivity, capturing 41.4% of true user churn profiles before they completely abandon the app platform.

###  Strategic Operational Insights for Product Teams
* **Velocity and Trip Patterns:** The feature importance scores show that `km_per_hour` is our strongest predictive signal. Users registering lower average speeds or predominantly completing short, local trips present our highest turnover rates. The growth team should target this segment with localized, short-distance navigation incentives.
* **The Onboarding Drop-Off Window:** Account tenure (`n_days_after_onboarding`) carries massive predictive weight. User retention is heavily decided within the first few weeks of setup. If a new user doesn't establish active daily driving habits during this initial onboarding window, they present an immediate churn risk. We need to focus on product features that drive early user engagement right after account creation.

###  Environment & Reproducibility Requirements
* **Core Handling:** `numpy`, `pandas`
* **Visualization:** `matplotlib`
* **Modeling Infrastructure:** `scikit-learn` (Random Forest, GridSearchCV)
* **Gradient Boosting:** `xgboost` (XGBClassifier)
* **Serialization Tool:** `pickle`

### 📄 License
This repository is licensed under the open-source **MIT License**—feel free to use, modify, and distribute the code baseline as needed.
