"""
main.py - Car Evaluation System
Single file solution: trains model in memory and provides prediction interface.
Only requires scikit-learn. No file I/O, no pickle, no pandas, no numpy.
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading data...")

# Define column names and categories
COLUMNS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
CATEGORIES = [
    ['vhigh', 'high', 'med', 'low'],
    ['vhigh', 'high', 'med', 'low'],
    ['2', '3', '4', '5more'],
    ['2', '4', 'more'],
    ['small', 'med', 'big'],
    ['low', 'med', 'high']
]

# Load data from car.data file
features = []
targets = []

try:
    with open('car.data', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = line.split(',')
                features.append(values[:-1])
                targets.append(values[-1])
except FileNotFoundError:
    print("[!!] ERROR: car.data file not found!")
    print("[+]Download from: https://archive.ics.uci.edu/dataset/19/car+evaluation")
    exit(1)

print(f"[!] Loaded {len(features)} samples")

# ============================================================================
# DATA ENCODING
# ============================================================================

print("[!] Encoding data...")

feature_encoder = OrdinalEncoder(categories=CATEGORIES)
target_encoder = LabelEncoder()

X = feature_encoder.fit_transform(features)
y = target_encoder.fit_transform(targets)

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("[!] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("[!] Training model...")

model = CategoricalNB()
model.fit(X_train, y_train)

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("[!] Evaluating model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"[!] Model accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# ============================================================================
# PREDICTION INTERFACE
# ============================================================================

print("\n" + "=" * 50)
print("CAR EVALUATION SYSTEM")
print("=" * 50)
print("[+] Attributes: buying, maint, doors, persons, lug_boot, safety")
print("\nValid values:")
for col, cats in zip(COLUMNS, CATEGORIES):
    print(f"\t{col}: {', '.join(cats)}")
print("\n[+] Example: vhigh,med,3,2,small,low")
print("[+] Commands: quit/exit/q (exit), help (show values)")
print("=" * 50)

# Prediction mapping
MAPPING = {
    'unacc': 'Unacceptable',
    'acc': 'Acceptable',
    'good': 'Good',
    'vgood': 'Very Good'
}

# Main loop
while True:
    user_input = input("\nEnter values: ").strip()
    
    # Handle exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    # Handle help command
    if user_input.lower() == 'help':
        print("\n[+] Valid values:")
        for col, cats in zip(COLUMNS, CATEGORIES):
            print(f"  {col}: {', '.join(cats)}")
        continue
    
    # Skip empty input
    if not user_input:
        continue
    
    # Parse and validate input
    values = [v.strip() for v in user_input.split(',')]
    
    if len(values) != 6:
        print(f"[!!] Error: Expected 6 values, got {len(values)}")
        continue
    
    valid = True
    for i, (val, col, cats) in enumerate(zip(values, COLUMNS, CATEGORIES)):
        if val not in cats:
            print(f"[!!] Error: '{val}' invalid for {col}")
            print(f"\tValid options: {', '.join(cats)}")
            valid = False
            break
    
    if not valid:
        continue
    
    # Make prediction
    encoded = feature_encoder.transform([values])
    pred_encoded = model.predict(encoded)[0]
    prediction = target_encoder.inverse_transform([pred_encoded])[0]
    
    # Show result
    print(f"\n[=] Prediction: {MAPPING.get(prediction, prediction)}")
    
    # Show confidence (probabilities)
    probs = model.predict_proba(encoded)[0]
    print("[+] Confidence:")
    for cls, prob in zip(target_encoder.classes_, probs):
        bar = "#" * int(prob * 20) + "-" * (20 - int(prob * 20))
        print(f"\t{MAPPING.get(cls, cls):12} {bar} {prob*100:.1f}%")
