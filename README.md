# Navigation User Retention Pipeline: Optimizing Platform Engagement

**Project Overview:** 
I am building an end-to-end predictive machine learning classification pipeline to identify user attrition patterns for a high-volume GPS navigation platform. The goal is to isolate the specific behavioral triggers and usage thresholds of departing user segments, providing product growth teams with an early-warning system to optimize engagement lifecycles and increase long-term user retention.[Navigation_User_Retention_Pipeline_Optimizing_Platform_Engagement.ipynb](Navigation_User_Retention_Pipeline_Optimizing_Platform_Engagement.ipynb)

### Model Evaluation & Decision Boundary Calibration

I tested high-capacity ensemble architectures across a stratified 60/20/20 data split derived from a database of 15,000 active app users. 

Because standard classification setups default to a 0.50 probability threshold, they failed to detect true user turnover thresholds, leading to highly conservative model sensitivity:
* The baseline Random Forest model hit an 81.8% accuracy but returned a very low Recall score of 12.6%.
* The baseline XGBoost model delivered an 80.9% accuracy and a 17.3% recall score.

In a consumer application retention project, missing a user who is preparing to abandon the platform (a False Negative) carries a significantly higher business cost than flagging a safe user (a False Positive). To optimize the pipeline for maximum real-world business value, I ran an automated grid tuning pass to shift our operational decision boundary down from 0.50 to **0.194**. 

This calibration successfully more than doubled our model's sensitivity on the final test set split:
* **Final Model Precision:** 33.5%
* **Final Model Recall:** 41.4%
* **Final Test Set Accuracy:** 75.0%

**Engineering Conclusion:** By accepting a manageable 5.9% drop in overall classification accuracy (from 80.9% down to 75.0%), we successfully more than doubled our model's sensitivity, capturing 41.4% of true user attrition profiles before they completely abandon the app platform.

### Core Data Insights & Strategic Product Recommendations

* **Velocity and Trip Patterns:** Feature importance mapping revealed that average trip velocity (`km_per_hour`) is our single strongest predictive signal. Users registering lower average speeds or predominantly completing short, local trips present our highest turnover rates. The product growth team should target this specific segment with localized, short-distance navigation incentives.
* **The Onboarding Habit Window:** Account tenure since onboarding (`n_days_after_onboarding`) carries massive predictive weight. User retention is heavily decided within the first few weeks of setup. If a new user does not establish active daily driving habits during this initial onboarding window, they present an immediate churn risk. We need to focus on product features that drive early user engagement right after account creation.

### Environment & Reproducibility Requirements
* **Data Pipelines:** `numpy`, `pandas`
* **Plotting & Visuals:** `matplotlib`
* **Modeling Pipelines:** `scikit-learn` (Random Forest, GridSearchCV)
* **Gradient Boosting:** `xgboost` (XGBClassifier)
* **Model Serialization:** `pickle`

### 📄 License
This repository is licensed under the open-source **MIT License**—feel free to use, modify, and distribute the code baseline as needed.


