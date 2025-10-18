import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('concrete_data.csv')
train_idx = list(range(0, 501)) + list(range(631, 1030))
test_idx  = list(range(501, 631))
train = df.iloc[train_idx]
test  = df.iloc[test_idx]
#Columns
X_cols = df.columns[:-1]
y_col  = df.columns[-1]
X_train = train[X_cols]
X_test  = test[X_cols]
y_train = train[y_col]
y_test  = test[y_col]

def regression(X_tr, X_te, y_tr, y_te, set_name):
    X_tr_const = sm.add_constant(X_tr)
    X_te_const = sm.add_constant(X_te)
    model = sm.OLS(y_tr, X_tr_const).fit()
    y_pred_train = model.predict(X_tr_const)
    y_pred_test  = model.predict(X_te_const)
    mse_train = mean_squared_error(y_tr, y_pred_train)
    mse_test  = mean_squared_error(y_te, y_pred_test)
    r2_train = model.rsquared
    r2_test  = 1 - ((y_te - y_pred_test)**2).sum() / ((y_te - y_te.mean())**2).sum()
    
    print(f"\n{set_name}:")
    print(f"MSE train: {mse_train:.4g}")
    print(f"MSE test:  {mse_test:.4g}")
    print(f"R^2 train:  {r2_train:.4g}")
    print(f"R^2 test:   {r2_test:.4g}")
    
    summary_df = pd.DataFrame({
        'Coefficient': model.params.round(4),
        't-stat': model.tvalues.round(4),
        'p-value': model.pvalues.apply(lambda x: f"{x:.3e}")
    })
    
    print("\nPer-coefficient statistical analysis:")
    print(summary_df)
    print("\n (F-stat) p-value:", f"{model.f_pvalue:.3e}")
    
    return model, mse_train, mse_test, r2_train, r2_test, summary_df

#set 1 standardized
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)
X_train_std_df = pd.DataFrame(X_train_std, columns=X_cols, index=X_train.index)
X_test_std_df  = pd.DataFrame(X_test_std, columns=X_cols, index=X_test.index)
regression(X_train_std_df, X_test_std_df, y_train, y_test, "Set 1: Standardized")
#set 2 raw
regression(X_train, X_test, y_train, y_test, "Set 2: Raw")
#set 3 log trans
X_train_log = np.log1p(X_train)
X_test_log  = np.log1p(X_test)
regression(X_train_log, X_test_log, y_train, y_test, "Set 3: Log-transformed")