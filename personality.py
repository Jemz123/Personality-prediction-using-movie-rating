import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from CSV file
file_path = r'C:\Users\Administrator\Desktop\pythonprojects\2018-personality-data.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Display column names and the first few rows of the dataset
print("Column Names in DataFrame:")
print(df.columns.tolist())

print("\nDataset Preview:")
print(df.head())

# Strip any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# List of expected feature columns
expected_features = [
    'movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5',
    'movie_6', 'movie_7', 'movie_8', 'movie_9', 'movie_10',
    'movie_11', 'movie_12', 'assigned metric', 'assigned condition'
]

# Check if all expected feature columns are present
missing_features = [col for col in expected_features if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing columns: {missing_features}")

# Select features and target variables
features = df[expected_features]

# Convert categorical features to dummy variables
features = pd.get_dummies(features, columns=['assigned metric', 'assigned condition'], drop_first=True)

# Target variables
targets = df[['openness', 'agreeableness', 'emotional_stability', 'conscientiousness', 'extraversion']]

# Initialize dictionary to store models
models = {}

# Train and evaluate models for each personality trait
for trait in targets.columns:
    print(f"\nTraining model for {trait}...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets[trait], test_size=0.25, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {trait}: {mse}")
    
    models[trait] = model

# Example: Predict personality traits for a new sample
new_sample = pd.DataFrame({
    'movie_1': [4.5],
    'movie_2': [3.2],
    'movie_3': [4.0],
    'movie_4': [5.0],
    'movie_5': [3.8],
    'movie_6': [4.2],
    'movie_7': [4.0],
    'movie_8': [3.6],
    'movie_9': [4.4],
    'movie_10': [4.1],
    'movie_11': [3.9],
    'movie_12': [4.3],
    'assigned metric': ['serendipity'],
    'assigned condition': ['high']
})

# Convert categorical features to dummy variables
new_sample = pd.get_dummies(new_sample, columns=['assigned metric', 'assigned condition'], drop_first=True)

# Ensure the new sample has all the columns used in training
for col in features.columns:
    if col not in new_sample.columns:
        new_sample[col] = 0

# Align column order
new_sample = new_sample[features.columns]

# Predict personality traits
predictions = {trait: model.predict(new_sample)[0] for trait, model in models.items()}
print("\nPersonality Predictions for the new sample:")
print(predictions)
