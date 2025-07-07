# confusion_matrix_analysis_v2.py
# Advanced confusion matrix analysis for signature verification system
# Optimized threshold analysis with enhanced visualizations

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Load JSON results file (New optimized threshold results)
base_dir = r"C:\\Users\\gulme\\OneDrive\\Desktop\\dataset"
json_path = os.path.join(base_dir, "results2.json")

print("Loading optimized JSON data (Threshold 0.30)...")
with open(json_path, "r", encoding="utf-8") as f:
    results = json.load(f)

# Extract ground truth and predictions
y_true = []  # Actual labels
y_pred = []  # Model predictions

for result in results:
    y_true.append(result["ground_truth_same"])
    y_pred.append(result["predicted_same"])

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate performance metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\nOptimized Confusion Matrix Results (Threshold 0.30):")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
print(f"F1-Score: {f1_score:.3f} ({f1_score*100:.1f}%)")

# Visualize confusion matrix with new color schemes
plt.figure(figsize=(14, 6))

# First subplot: Confusion Matrix (counts) - Orange theme
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix v2.0 (Counts)\nThreshold = 0.30', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')

# Second subplot: Normalized Confusion Matrix (percentages) - Purple theme
plt.subplot(1, 2, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'],
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix v2.0 (%)\nThreshold = 0.30', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')

plt.tight_layout()

# Save visualization
viz_path = os.path.join(base_dir, "confusion_matrix_v2.png")
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nOptimized confusion matrix saved: {viz_path}")

# Threshold analysis with different threshold values
print("\nAdvanced Threshold Analysis:")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
threshold_results = []

for thresh in thresholds:
    y_pred_thresh = [result["match_ratio"] >= thresh for result in results]
    cm_thresh = confusion_matrix(y_true, y_pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
    acc_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
    
    threshold_results.append({
        'threshold': thresh,
        'accuracy': acc_t,
        'tp': tp_t,
        'tn': tn_t,
        'fp': fp_t,
        'fn': fn_t
    })
    
    print(f"Threshold {thresh:.2f}: Accuracy = {acc_t:.3f} ({acc_t*100:.1f}%)")

# Find optimal threshold
best_threshold = max(threshold_results, key=lambda x: x['accuracy'])
print(f"\n🏆 Optimal threshold: {best_threshold['threshold']:.2f} (Accuracy: {best_threshold['accuracy']:.3f})")

# Threshold vs Accuracy visualization - NEW DESIGN!
plt.figure(figsize=(12, 7))
threshs = [r['threshold'] for r in threshold_results]
accs = [r['accuracy'] for r in threshold_results]

# Gradient color effect
colors = plt.cm.viridis(np.linspace(0, 1, len(threshs)))
plt.scatter(threshs, accs, c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=2)
plt.plot(threshs, accs, 'r-', linewidth=3, alpha=0.7)

plt.xlabel('Threshold Value', fontweight='bold', fontsize=12)
plt.ylabel('Accuracy Score', fontweight='bold', fontsize=12)
plt.title('Threshold Optimization Analysis v2.0\nSignature Verification System', 
          fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')

# Highlight optimal threshold
plt.axvline(x=best_threshold['threshold'], color='gold', linestyle='--', linewidth=3,
            label=f'Optimal: {best_threshold["threshold"]:.2f} ({best_threshold["accuracy"]:.1%})')
plt.axhline(y=best_threshold['accuracy'], color='gold', linestyle='--', linewidth=2, alpha=0.7)

# Add annotations for each point
for i, (thresh, acc) in enumerate(zip(threshs, accs)):
    plt.annotate(f'{acc:.1%}', (thresh, acc), textcoords="offset points", 
                xytext=(0,10), ha='center', fontweight='bold', fontsize=10)

plt.legend(fontsize=12, framealpha=0.9)
plt.ylim(0.85, 1.0)

# Save threshold analysis
thresh_path = os.path.join(base_dir, "threshold_analysis_v2.png")
plt.savefig(thresh_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Optimized threshold analysis saved: {thresh_path}")

plt.show()

# Detailed performance report
print(f"\nDetailed Performance Report:")
print("="*50)
print(f"OPTIMIZED SYSTEM PERFORMANCE (Threshold 0.30)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=['Different Person', 'Same Person']))

# Calculate improvement (compare with old results.json)
try:
    old_json_path = os.path.join(base_dir, "results.json")
    with open(old_json_path, "r", encoding="utf-8") as f:
        old_results = json.load(f)
    
    old_y_true = [r["ground_truth_same"] for r in old_results]
    old_y_pred = [r["predicted_same"] for r in old_results]
    old_cm = confusion_matrix(old_y_true, old_y_pred)
    old_tn, old_fp, old_fn, old_tp = old_cm.ravel()
    old_accuracy = (old_tp + old_tn) / (old_tp + old_tn + old_fp + old_fn)
    
    improvement = (accuracy - old_accuracy) * 100
    print(f"\nIMPROVEMENT ANALYSIS:")
    print(f"Previous Accuracy (0.10): {old_accuracy:.3f} ({old_accuracy*100:.1f}%)")
    print(f"Current Accuracy (0.30): {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Performance Improvement: +{improvement:.1f} percentage points!")
    
except:
    print(f"\nPrevious results.json not found, comparison unavailable.")

print("\nAdvanced analysis completed!")
print("Enhanced with new color schemes and professional visualizations!")
