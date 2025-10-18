import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.m = None
        self.b = None
        self.losses = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Initialize parameters
        if len(X.shape) == 1:
            self.m = 0.0
            self.b = 0.0
        else:
            n_features = X.shape[1]
            self.m = np.zeros(n_features)
            self.b = 0.0
        
        # Gradient descent
        for i in range(self.n_iterations):
            if len(X.shape) == 1:
                y_pred = self.m * X + self.b
            else:
                y_pred = np.dot(X, self.m) + self.b
            
            # Calculate loss
            mse = np.mean((y - y_pred) ** 2)
            self.losses.append(mse)
            
            # Calculate gradients
            if len(X.shape) == 1:
                dm = (2/n_samples) * np.sum(X * (y_pred - y))
                db = (2/n_samples) * np.sum(y_pred - y)
                
                # Update parameters
                self.m -= self.lr * dm
                self.b -= self.lr * db
            else:
                dm = (2/n_samples) * np.dot(X.T, (y_pred - y))
                db = (2/n_samples) * np.sum(y_pred - y)
                
                # Update parameters
                self.m -= self.lr * dm
                self.b -= self.lr * db
                
    def predict(self, X):
        if len(X.shape) == 1:
            return self.m * X + self.b
        else:
            return np.dot(X, self.m) + self.b
    
    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def r_squared(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

def normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)

def standardize(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    return (X - X_mean) / (X_std + 1e-8)

df = pd.read_excel('Concrete_Data.xls')

feature_columns = df.columns[:-1].tolist()
target_column = df.columns[-1]

print(f"\nFeatures: {feature_columns}")
print(f"Target: {target_column}")

# Split data
test_indices = range(501, 631)
train_indices = list(range(0, 501)) + list(range(631, len(df)))

X_train = df.iloc[train_indices, :-1].values
y_train = df.iloc[train_indices, -1].values
X_test = df.iloc[test_indices, :-1].values
y_test = df.iloc[test_indices, -1].values

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Q1.1
print("Q1.1:")

X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

results_q1_1 = {}

# learning rates
learning_rates_norm = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i, feature in enumerate(feature_columns):
    print(f"\n{feature} as predictor:")
    
    model = LinearRegression(learning_rate=learning_rates_norm[i], n_iterations=10000)
    model.fit(X_train_norm[:, i], y_train)
    
    # metrics
    train_mse = model.mse(X_train_norm[:, i], y_train)
    train_r2 = model.r_squared(X_train_norm[:, i], y_train)
    test_mse = model.mse(X_test_norm[:, i], y_test)
    test_r2 = model.r_squared(X_test_norm[:, i], y_test)
    
    print(f"m = {model.m:.6f}, b = {model.b:.6f}")
    print(f"MSE on training data: {train_mse:.4f}")
    print(f"Variance Explained / R-Squared on training data: {train_r2:.4f}")
    print(f"MSE on testing data: {test_mse:.4f}")
    print(f"Variance Explained / R-Squared on testing data: {test_r2:.4f}")
    
    results_q1_1[feature] = {
        'm': model.m, 'b': model.b,
        'train_mse': train_mse, 'train_r2': train_r2,
        'test_mse': test_mse, 'test_r2': test_r2
    }

# Q1.2
print("Q1.2")
results_q1_2 = {}

# learning rates
learning_rates_raw = [0.00001, 0.000001, 0.000001, 0.0001, 0.0001, 0.0000001, 0.0000001, 0.0001]

for i, feature in enumerate(feature_columns):
    print(f"\n{feature} as predictor:")
    
    model = LinearRegression(learning_rate=learning_rates_raw[i], n_iterations=10000)
    model.fit(X_train[:, i], y_train)
    
    train_mse = model.mse(X_train[:, i], y_train)
    train_r2 = model.r_squared(X_train[:, i], y_train)
    test_mse = model.mse(X_test[:, i], y_test)
    test_r2 = model.r_squared(X_test[:, i], y_test)
    
    print(f"m = {model.m:.6f}, b = {model.b:.6f}")
    print(f"MSE on training data: {train_mse:.4f}")
    print(f"Variance Explained / R-Squared on training data: {train_r2:.4f}")
    print(f"MSE on testing data: {test_mse:.4f}")
    print(f"Variance Explained / R-Squared on testing data: {test_r2:.4f}")
    
    results_q1_2[feature] = {
        'm': model.m, 'b': model.b,
        'train_mse': train_mse, 'train_r2': train_r2,
        'test_mse': test_mse, 'test_r2': test_r2
    }

# Q2.1 & Q2.2
print("Q2:")

# Q2.1: Single sample update
print("\nQ2.1:")
model_test = LinearRegression(learning_rate=0.1, n_iterations=1)
model_test.m = np.array([1.0, 1.0, 1.0])
model_test.b = 1.0

X_test_single = np.array([[3, 4, 5]])
y_test_single = np.array([4])
model_test.fit(X_test_single, y_test_single)

print(f"New m_1: {model_test.m[0]:.6f}")
print(f"New m_2: {model_test.m[1]:.6f}")
print(f"New m_3: {model_test.m[2]:.6f}")
print(f"New b: {model_test.b:.6f}")

# Q2.2
print("\nQ2.2")
model_test2 = LinearRegression(learning_rate=0.1, n_iterations=1)
model_test2.m = np.array([1.0, 1.0, 1.0])
model_test2.b = 1.0

X_test_multi = np.array([[3, 4, 4], [4, 2, 1], [10, 2, 5], [3, 4, 5], [11, 1, 1]])
y_test_multi = np.array([3, 2, 8, 4, 5])
model_test2.fit(X_test_multi, y_test_multi)

print(f"New m_1: {model_test2.m[0]:.6f}")
print(f"New m_2: {model_test2.m[1]:.6f}")
print(f"New m_3: {model_test2.m[2]:.6f}")
print(f"New b: {model_test2.b:.6f}")

# Q2.3
print("Q2.3: Multivariate Linear Regression with Normalized Features")

model_multi_norm = LinearRegression(learning_rate=0.1, n_iterations=10000)
model_multi_norm.fit(X_train_norm, y_train)

train_mse = model_multi_norm.mse(X_train_norm, y_train)
train_r2 = model_multi_norm.r_squared(X_train_norm, y_train)
test_mse = model_multi_norm.mse(X_test_norm, y_test)
test_r2 = model_multi_norm.r_squared(X_test_norm, y_test)

print(f"m values: {[round(m, 6) for m in model_multi_norm.m]}")
print(f"b value: {model_multi_norm.b:.6f}")
print(f"MSE on training data: {train_mse:.4f}")
print(f"Variance Explained / R-Squared on training data: {train_r2:.4f}")
print(f"MSE on testing data: {test_mse:.4f}")
print(f"Variance Explained / R-Squared on testing data: {test_r2:.4f}")

# Q2.4
print("Q2.4")

# standardize and convert
X_train_std = standardize(X_train)
X_test_std = standardize(X_test)

model_multi_raw = LinearRegression(learning_rate=0.01, n_iterations=10000)
model_multi_raw.fit(X_train_std, y_train)

std_devs = X_train.std(axis=0)
means = X_train.mean(axis=0)
raw_m = model_multi_raw.m / (std_devs + 1e-8)
raw_b = model_multi_raw.b - np.sum(raw_m * means)

# preds with raw data
y_pred_train = np.dot(X_train, raw_m) + raw_b
y_pred_test = np.dot(X_test, raw_m) + raw_b

train_mse = np.mean((y_train - y_pred_train) ** 2)
train_r2 = 1 - (np.sum((y_train - y_pred_train) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
test_mse = np.mean((y_test - y_pred_test) ** 2)
test_r2 = 1 - (np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f"m values: {[round(m, 6) for m in raw_m]}")
print(f"b value: {raw_b:.6f}")
print(f"MSE on training data: {train_mse:.4f}")
print(f"Variance Explained / R-Squared on training data: {train_r2:.4f}")
print(f"MSE on testing data: {test_mse:.4f}")
print(f"Variance Explained / R-Squared on testing data: {test_r2:.4f}")

print("Models with Positive Variance Explained")

print("\nQ1.1 (Normalized features) - Models with positive test VE:")
for feature, results in results_q1_1.items():
    if results['test_r2'] > 0:
        print(f"  - {feature}: Test VE = {results['test_r2']:.4f}")

print("\nQ1.2 (Raw features) - Models with positive test VE:")
for feature, results in results_q1_2.items():
    if results['test_r2'] > 0:
        print(f"  - {feature}: Test VE = {results['test_r2']:.4f}")