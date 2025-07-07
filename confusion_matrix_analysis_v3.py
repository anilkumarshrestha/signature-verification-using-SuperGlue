# confusion_matrix_analysis_v3.py
# Advanced confusion matrix analysis for signature verification system
# Uses results_v2_match01.json data (match_threshold=0.1 test)

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Load JSON results file (results_v2_match01.json)
base_dir = r"C:\\Users\\gulme\\OneDrive\\Desktop\\dataset"
json_path = os.path.join(base_dir, "results_v2_match01.json")

print("Loading results_v2_match01.json data (match_threshold=0.1)...")
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

print("\nConfusion Matrix Results v3.0 (match_threshold=0.1):")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
print(f"F1-Score: {f1_score:.3f} ({f1_score*100:.1f}%)")

# Visualize confusion matrix with NEW color schemes (v3)
plt.figure(figsize=(15, 7))

# First subplot: Confusion Matrix (counts) - Blue theme (YENİ!)
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix v3.0 (Counts)\nmatch_threshold=0.1 Analysis', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')

# Second subplot: Normalized Confusion Matrix (percentages) - Green theme (YENİ!)
plt.subplot(1, 2, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'],
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix v3.0 (%)\nmatch_threshold=0.1 Analysis', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')

plt.tight_layout()

# Save visualization
viz_path = os.path.join(base_dir, "confusion_matrix_v3.png")
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nConfusion matrix v3 saved: {viz_path}")

# Threshold analysis with different threshold values
print("\nAdvanced Threshold Analysis v3.0:")
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
print(f"\nOptimal threshold: {best_threshold['threshold']:.2f} (Accuracy: {best_threshold['accuracy']:.3f})")

# Threshold vs Accuracy visualization - NEW DESIGN v3!
plt.figure(figsize=(13, 8))
threshs = [r['threshold'] for r in threshold_results]
accs = [r['accuracy'] for r in threshold_results]

# NEW: Gradient color effect with plasma colormap
colors = plt.cm.plasma(np.linspace(0, 1, len(threshs)))
plt.scatter(threshs, accs, c=colors, s=120, alpha=0.9, edgecolors='navy', linewidth=2.5)
plt.plot(threshs, accs, 'darkblue', linewidth=4, alpha=0.8)

plt.xlabel('Threshold Value', fontweight='bold', fontsize=14)
plt.ylabel('Accuracy Score', fontweight='bold', fontsize=14)
plt.title('Threshold Optimization Analysis v3.0\nSignature Verification System (match_threshold=0.1)', 
          fontsize=18, fontweight='bold', pad=25)
plt.grid(True, alpha=0.4, linestyle=':', color='gray')

# Highlight optimal threshold - NEW STYLE
plt.axvline(x=best_threshold['threshold'], color='crimson', linestyle='--', linewidth=4,
            label=f'Optimal: {best_threshold["threshold"]:.2f} ({best_threshold["accuracy"]:.1%})')
plt.axhline(y=best_threshold['accuracy'], color='crimson', linestyle='--', linewidth=3, alpha=0.6)

# Add annotations for each point - NEW STYLE
for i, (thresh, acc) in enumerate(zip(threshs, accs)):
    plt.annotate(f'{acc:.1%}', (thresh, acc), textcoords="offset points", 
                xytext=(0,15), ha='center', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

plt.legend(fontsize=13, framealpha=0.9, loc='lower right')
plt.ylim(0.85, 1.0)

# Save threshold analysis
thresh_path = os.path.join(base_dir, "threshold_analysis_v3.png")
plt.savefig(thresh_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Threshold analysis v3 saved: {thresh_path}")

plt.show()

# Detailed performance report
print(f"\nDetailed Performance Report v3.0:")
print("="*60)
print(f"SYSTEM PERFORMANCE (match_threshold=0.1)")
print("="*60)
print(classification_report(y_true, y_pred, target_names=['Different Person', 'Same Person']))

# Compare with previous versions
try:
    # Compare with v2 results
    v2_json_path = os.path.join(base_dir, "results2.json")
    with open(v2_json_path, "r", encoding="utf-8") as f:
        v2_results = json.load(f)
    
    v2_y_true = [r["ground_truth_same"] for r in v2_results]
    v2_y_pred = [r["predicted_same"] for r in v2_results]
    v2_cm = confusion_matrix(v2_y_true, v2_y_pred)
    v2_tn, v2_fp, v2_fn, v2_tp = v2_cm.ravel()
    v2_accuracy = (v2_tp + v2_tn) / (v2_tp + v2_tn + v2_fp + v2_fn)
    
    improvement = (accuracy - v2_accuracy) * 100
    print(f"\nVERSION COMPARISON:")
    print(f"Results2.json (match_threshold=0.2): {v2_accuracy:.3f} ({v2_accuracy*100:.1f}%)")
    print(f"Results_v2_match01.json (match_threshold=0.1): {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if improvement > 0:
        print(f"Performance Improvement: +{improvement:.1f} percentage points!")
    elif improvement < 0:
        print(f"Performance Decrease: {improvement:.1f} percentage points")
    else:
        print(f"Performance Same: No change")
    
except:
    print(f"\nPrevious results2.json not found, comparison unavailable.")

print("\nAnalysis v3.0 completed!")
