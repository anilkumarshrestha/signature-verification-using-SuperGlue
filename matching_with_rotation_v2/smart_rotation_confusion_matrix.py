import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Load JSON results file
base_dir = r"C:\\Users\\gulme\\OneDrive\\Desktop\\dataset"
json_path = os.path.join(base_dir, "results_smart_rotation2.json")

print("Loading SMART rotation results...")
try:
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"✓ Loaded {len(results)} results")
except FileNotFoundError:
    print(f"✗ File not found: {json_path}")
    exit(1)

# Extract predictions
y_true = [r["ground_truth_same"] for r in results]
y_pred = [r["predicted_same"] for r in results]

# Calculate confusion matrix and metrics
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Display results
print("\n" + "="*50)
print("SMART ROTATION CONFUSION MATRIX RESULTS")
print("="*50)
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")
print(f"\nAccuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"Recall:    {recall:.3f} ({recall*100:.1f}%)")
print(f"F1-Score:  {f1_score:.3f} ({f1_score*100:.1f}%)")

# Rotation usage stats
rotation_used = sum(1 for r in results if r.get('rotation_used', False))
print(f"\nRotation used in {rotation_used}/{len(results)} cases ({rotation_used/len(results)*100:.1f}%)")

# Visualize confusion matrix
plt.figure(figsize=(12, 5))

# Counts
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'])
plt.title('Confusion Matrix (Counts)\nSMART Rotation System', fontweight='bold')

# Percentages
plt.subplot(1, 2, 2)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'])
plt.title('Confusion Matrix (%)\nSMART Rotation System', fontweight='bold')

plt.tight_layout()

# Save and show
save_path = os.path.join(base_dir, "smart_rotation_matrix_2.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved: {save_path}")
plt.show()

print("\nAnalysis completed!")
