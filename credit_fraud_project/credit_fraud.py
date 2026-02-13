# ============================================================
# Transaction Fraud Risk Analysis Project
# Author: Rafael Severiano
# ============================================================

# In this project, we build a fraud prediction model whose goal is not only to classify
# transactions, but to support decision-making. For each transaction, the model can
# recommend one of three actions: approve, block, or investigate.
#
# We explicitly model the cost of incorrect decisions: approving a fraudulent transaction,
# blocking an honest transaction, and investigating an honest transaction. While blocking
# or investigating legitimate customers carries a meaningful operational and customer cost,
# this cost is assumed to be lower than the cost of approving a fraudulent transaction.
# The absolute cost values are hypothetical and are used only to preserve realistic
# proportions between different types of errors.
#
# Since fraudulent transactions are rare and false positives lead to non-negligible costs,
# we initially prioritize precision during model training in order to avoid an overly
# punitive model that would unnecessarily block or investigate honest customers.
#
# Rather than aggressively maximizing recall at the training stage, we adjust decision
# thresholds after training and introduce an investigation rate, which defines a more
# conservative region between approval and blocking. By varying both thresholds and
# investigation rates, we evaluate more permissive and more conservative decision policies
# and select the configuration that minimizes the total expected cost.


# We start by defining the costs and a function to compute the total cost

cost_approve_fraud_transaction = 5000
cost_investigate_honest_transaction = 1000
cost_block_honest_transaction = 1200

def cost(number_frauds_approved,
         number_honest_transactions_investigated,
         number_honest_transactions_blocked):
    return (
        number_frauds_approved * cost_approve_fraud_transaction
        + number_honest_transactions_investigated * cost_investigate_honest_transaction
        + number_honest_transactions_blocked * cost_block_honest_transaction
    )

# ============================================================
# Imports
# ===========================================================

from sklearn.compose import ColumnTransformer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from pathlib import Path

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# Loading Data + EDA
# ===========================================================

df = pd.read_csv("data/creditcard.csv")
              
# Analyze class imbalance

plt.figure(figsize=(8, 5))
sns.countplot(x=df['Class'])
plt.yscale('log')
plt.ylabel('Number of Transactions')
plt.xlabel('Class')
plt.xticks([0, 1], ['No Fraud', 'Fraud'])
plt.title('Proportion of Fraud vs. No Fraud (Full Dataset)')
plt.savefig(FIG_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
plt.tight_layout()
plt.show()


# Since fraudulent transactions tend to exhibit abnormal and highly specific
# behavioral patterns, generating synthetic samples using SMOTE may create
# unrealistic interpolations between minority class instances.
#
# This can blur the decision boundary between fraudulent and legitimate
# transactions, potentially introducing noise and distorting the original
# data distribution.
#
# Therefore, instead of applying oversampling techniques, we preserve the
# natural class imbalance and focus on 
# hyperparameter tuning, and appropriate evaluation metrics.

# Dataset Analysis

print(df.info())
print(df.describe())
print(df.isna().sum())  

# No missing values, and the numeric columns, except for 'Time' and 'Amount', have already been handled via PCA.
# Therefore, we will only scale the 'Time' and 'Amount' columns. Let's explore how these features behave to choose the best scaling method.


plt.figure()
plt.boxplot(df['Time'])
plt.xticks([1], ['Time Boxplot'])
plt.tight_layout()
plt.savefig(FIG_DIR / "time_boxplot.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure()
plt.boxplot(df['Amount'])
plt.yscale('log')
plt.xticks([1], ['Amount Boxplot'])
plt.tight_layout()
plt.savefig(FIG_DIR / "amount_boxplot.png", dpi=200, bbox_inches="tight")
plt.show()

# The 'Amount' column has many outliers, so we will use RobustScaler.

# ============================================================
# Data Separation
# ============================================================

X = df.drop('Class', axis='columns')
y = df['Class']


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=0
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=0
)

# ============================================================
# Preprocessing
# ============================================================

t_time = ('time', StandardScaler(), ['Time'])
t_amount = ('amount', RobustScaler(), ['Amount'])
preprocessor = ColumnTransformer(transformers=[t_time, t_amount], remainder='passthrough')

# ============================================================
# Hyperparameter Tuning
# ============================================================

model_log = Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=1000))])
params = {'model__C': [0.001, 0.01, 0.1, 0.5], 'model__class_weight': [{0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}]}
cv = StratifiedKFold(n_splits=5)
gs_log = GridSearchCV(model_log, params, scoring='precision', cv=cv)
gs_log.fit(X_train, y_train)

# ============================================================
# Choosing Thresholds and Investigation Rates
# ============================================================

#We are building a model that prioritizes precision, aiming to minimize false positives. 
#The risk of this approach is that the model may become too permissive toward fraudulent transactions. 
#Our goal is to explore different configurations to identify the threshold and investigation rate 
#that minimize the total cost of actions (block, investigate, approve). 
#We select candidate threshold values using the precision–recall curve, testing different conservative levels.

limits =[0.01, 0.1, 0.4, 0.6, 0.9, 0.99]
thresholds_test = []
y_val_proba = gs_log.best_estimator_.predict_proba(X_val)[:, 1]

# Evaluate precision-recall curve for threshold tuning
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
ap_score = average_precision_score(y_val, y_val_proba)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AP = {ap_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(FIG_DIR / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
plt.show()

for precision_min in limits:
    idx_valids = np.where(precision >= precision_min)[0]
    if len(idx_valids) > 0:  
        best_idx = idx_valids[np.argmax(recall[idx_valids])]
        best_threshold = thresholds[best_idx]
        thresholds_test.append(best_threshold)


investigation_rate_levels = [0.90,0.8,0.5,1 ]  

# Choosing 1 means the model does not investigate; it only blocks or approves.

# ============================================================
# Calculations and Plots (Validation Set and Test Set)
# ============================================================

real_frauds_val = sum(y_val)  # Total number of frauds in the validation set
honest_transactions_val = len(y_val) - real_frauds_val  # Total number of honest transactions in the validation set
real_frauds_test = sum(y_test)  # Total number of frauds in the test set
honest_transactions_test = len(y_test) - real_frauds_test  # Total number of honest transactions in the test set


y_test_proba = gs_log.best_estimator_.predict_proba(X_test)[:, 1]


df_fraud_val = pd.DataFrame([])
df_fraud_test = pd.DataFrame([])
df_honest_val = pd.DataFrame([])
df_honest_test = pd.DataFrame([])
 

#We include the original threshold 
#We include 0.70 to test a threshold that is permissive, but not excessively so.


thresholds_test=thresholds_test+[0.5,0.70]

for best_threshold in thresholds_test:
    for investigation_rate in investigation_rate_levels:  
        actions_val = [] 
        actions_test = []
        for i in range(len(y_val_proba)):
            if y_val_proba[i] >= best_threshold:
                actions_val.append('Block')
            elif y_val_proba[i] < best_threshold and y_val_proba[i] >= best_threshold * investigation_rate:
                actions_val.append('Investigate')
            else:
                actions_val.append('Approve')
        for i in range(len(y_test_proba)):
            if y_test_proba[i] >= best_threshold:
                actions_test.append('Block')
            elif y_test_proba[i] < best_threshold and y_test_proba[i] >= best_threshold * investigation_rate:
                actions_test.append('Investigate')
            else:
                actions_test.append('Approve')
        blocked_frauds_val = 0
        investigated_frauds_val = 0
        missed_frauds_val = 0
        blocked_frauds_test = 0
        investigated_frauds_test = 0
        missed_frauds_test = 0
        for i in range(len(actions_val)):
            if actions_val[i] == 'Block' and y_val.iloc[i] == 1:
                blocked_frauds_val += 1
            elif actions_val[i] == 'Investigate' and y_val.iloc[i] == 1:
                investigated_frauds_val += 1
            elif actions_val[i] == 'Approve' and y_val.iloc[i] == 1:
                missed_frauds_val += 1
        for i in range(len(actions_test)):
            if actions_test[i] == 'Block' and y_test.iloc[i] == 1:
                blocked_frauds_test += 1
            elif actions_test[i] == 'Investigate' and y_test.iloc[i] == 1:
                investigated_frauds_test += 1
            elif actions_test[i] == 'Approve' and y_test.iloc[i] == 1:
                missed_frauds_test += 1
        incorrectly_blocked_transactions_val = 0
        incorrectly_investigated_transactions_val = 0
        correctly_approved_transactions_val = 0
        incorrectly_blocked_transactions_test = 0
        incorrectly_investigated_transactions_test = 0
        correctly_approved_transactions_test = 0
        for i in range(len(actions_val)):
            if actions_val[i] == 'Block' and y_val.iloc[i] == 0:
                incorrectly_blocked_transactions_val += 1
            elif actions_val[i] == 'Investigate' and y_val.iloc[i] == 0:
                incorrectly_investigated_transactions_val += 1
            elif actions_val[i] == 'Approve' and y_val.iloc[i] == 0:
                correctly_approved_transactions_val += 1
        for i in range(len(actions_test)):
            if actions_test[i] == 'Block' and y_test.iloc[i] == 0:
                incorrectly_blocked_transactions_test += 1
            elif actions_test[i] == 'Investigate' and y_test.iloc[i] == 0:
                incorrectly_investigated_transactions_test += 1
            elif actions_test[i] == 'Approve' and y_test.iloc[i] == 0:
                correctly_approved_transactions_test += 1
        cost_value_val = cost(missed_frauds_val, incorrectly_investigated_transactions_val, incorrectly_blocked_transactions_val)
        cost_value_test = cost(missed_frauds_test, incorrectly_investigated_transactions_test, incorrectly_blocked_transactions_test)

       
        # Fraud transaction analysis for the validation set
     
        values_fraud = [real_frauds_val, blocked_frauds_val, investigated_frauds_val, missed_frauds_val, cost_value_val]
        df_fraud_val[f"({best_threshold}, {investigation_rate})"]=values_fraud
        
        # Honest transaction analysis for the validation set
        
        values_honest = [honest_transactions_val, incorrectly_blocked_transactions_val, incorrectly_investigated_transactions_val, correctly_approved_transactions_val, cost_value_val]
        df_honest_val[f"({best_threshold}, {investigation_rate})"]=values_honest
        
        # Fraud transaction analysis for the test set
    
        values_fraud = [real_frauds_test, blocked_frauds_test, investigated_frauds_test, missed_frauds_test, cost_value_test]
        df_fraud_test[f"({best_threshold}, {investigation_rate})"]=values_fraud
        
        # Honest transaction analysis for the test set
        values_honest = [honest_transactions_test, incorrectly_blocked_transactions_test, incorrectly_investigated_transactions_test, correctly_approved_transactions_test, cost_value_test]
        df_honest_test[f"({best_threshold}, {investigation_rate})"]=values_honest 
        
labels_fraud = ['Real Fraud', 'Blocked Fraud', 'Investigated Fraud', 'Missed Fraud', 'Cost']        
labels_honest = ['Honest Transactions', 'Incorrectly Blocked', 'Incorrectly Investigated', 'Correctly Approved', 'Cost']    

df_fraud_test.index = labels_fraud
df_fraud_val.index = labels_fraud
df_honest_test.index = labels_honest
df_honest_val.index = labels_honest 

plt.figure(figsize=(15, 10))
sns.heatmap(df_fraud_val.T.sort_values(by=['Cost'], ascending=True), cmap='Blues', annot=True, fmt='d')
plt.title('Fraud Transactions Analysis (Validation Set)', fontsize=10)
plt.ylabel('Threshold X Investigation Rate', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "fraud_transactions_analysis_validation_set.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(df_honest_val.T.sort_values(by=['Cost'], ascending=True), cmap='Blues', annot=True, fmt='d')
plt.title('Honest Transactions Analysis (Validation Set)', fontsize=10)
plt.ylabel('Threshold X Investigation Rate', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "honest_transactions_analysis_validation_set.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(df_fraud_test.T.sort_values(by=['Cost'], ascending=True), cmap='Blues', annot=True, fmt='d')
plt.title('Fraud Transactions Analysis (Test Set)', fontsize=10)
plt.ylabel('(Threshold X Investigation Rate)', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "fraud_transactions_analysis_test_set.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(df_honest_test.T.sort_values(by=['Cost'], ascending=True), cmap='Blues', annot=True, fmt='d')
plt.title('Honest Transactions Analysis (Test Set)', fontsize=10)
plt.ylabel('Threshold X Investigation Rate', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "honest_transactions_analysis_test_set.png", dpi=200, bbox_inches="tight")
plt.show()

# The model shows stable behavior across a wide range of decision thresholds. Even at lower thresholds,
# false positives do not increase disproportionately, while at higher thresholds the model still captures
# a significant portion of fraudulent transactions. This suggests a meaningful separation between
# predicted probability distributions of the two classes.
#
# We will now examine metrics and visualizations to further assess the model's performance and confirm
# the separation.


# ============================================================
# Analyzing the model
# ============================================================

# Receiver Operating Characteristic (ROC) Curve

fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR / "roc.png", dpi=200, bbox_inches="tight")
plt.show()

fraud_probs = y_val_proba[y_val == 1]
non_fraud_probs = y_val_proba[y_val == 0]

# Probability Distribution for Fraud vs. Non-Fraud

plt.figure(figsize=(10, 6))
sns.kdeplot(fraud_probs, color='red', label='Fraud', shade=True, alpha=0.6)
sns.kdeplot(non_fraud_probs, color='blue', label='Non-Fraud', shade=True, alpha=0.6)

plt.title('Probability Distribution for Fraud vs. Non-Fraud (Validation Set)', fontsize=14)
plt.xlabel('Predicted Probability of Fraud', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(FIG_DIR / "probability_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

# The AUC, AP and probability distribution graph confirm our expectations.

#Conclusion:
# The model was initially trained with a strong focus on precision, with the goal of avoiding an excessively punitive policy in a highly imbalanced classification setting. 
# As a direct consequence of this choice, the model learns a clear separation in the probabilistic space: the vast majority of honest transactions receive very low fraud probabilities, 
# while fraudulent transactions concentrate at relatively higher probability values. This separation reduces the overlap between the distributions and creates a safety region 
# in which threshold adjustments can be made in a controlled manner.
# This behavior explains why more conservative fraud policies did not result in a significant increase in false positives. Even with substantial reductions in the decision threshold, 
# most honest transactions remain outside the blocking and investigation regions. As a result, lowering the threshold proves to be a valid approach in this specific context, allowing 
# additional frauds to be captured without incurring excessive costs from unnecessary blocks or investigations.
# On the other hand, highly permissive fraud policies also showed relatively good performance. This occurs primarily because fraud events are extremely rare, meaning that approving 
# most transactions already constitutes a competitive baseline policy. In addition, the strong separation between classes in the probability space ensures that indiscriminate approval 
# does not lead to a disproportionate number of approved fraudulent transactions, thereby limiting the cost associated with this type of error.
#
# The results indicate that the best performance is achieved by an intermediate policy that preserves the base model and introduces only a moderate additional level of conservatism 
# through the investigation region. This strategy efficiently balances the reduction of frauds incorrectly approved with the control of operational costs and customer friction. 
# Importantly, this choice is not supported solely by the validation set. The test set confirms the same performance pattern, reinforcing the robustness 
# of the decision and the absence of opportunistic threshold tuning.
#
# Finally, extremely conservative policies characterized by high thresholds and wide blocking or investigation regions show inferior performance. This outcome is consistent with the 
# rarity of fraud events and with the costs associated with intervening in honest transactions, where increased customer friction outweighs the marginal gains from further fraud reduction.



