import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

base_dir = r"C:\\Users\\gulme\\OneDrive\\Desktop\\dataset"
json_path = os.path.join(base_dir, "results.json")

print("JSON verilerini yüklüyor...")
with open(json_path, "r", encoding="utf-8") as f:
    results = json.load(f)

y_true = []  # Gerçek etiketler
y_pred = []  # Model tahminleri

for result in results:
    y_true.append(result["ground_truth_same"])
    y_pred.append(result["predicted_same"])

# Confusion matrix hesapla
cm = confusion_matrix(y_true, y_pred)

# Metrikleri hesapla
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\nConfusion Matrix Sonuçları:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"\nPerformans Metrikleri:")
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
print(f"F1-Score: {f1_score:.3f} ({f1_score*100:.1f}%)")

# Confusion matrix görselleştir
plt.figure(figsize=(12, 5))

# İlk subplot: Confusion Matrix (sayılar)
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'])
plt.title('Confusion Matrix (Counts)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# İkinci subplot: Normalized Confusion Matrix (yüzdeler)
plt.subplot(1, 2, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=['Predicted: Different', 'Predicted: Same'],
            yticklabels=['Actual: Different', 'Actual: Same'])
plt.title('Normalized Confusion Matrix (%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()

# Görseli kaydet
viz_path = os.path.join(base_dir, "confusion_matrix.png")
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix kaydedildi: {viz_path}")

# Threshold analizi için farklı eşik değerleri dene
print("\nThreshold Analizi:")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
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

# En iyi threshold'u bul
best_threshold = max(threshold_results, key=lambda x: x['accuracy'])
print(f"\nEn iyi threshold: {best_threshold['threshold']:.2f} (Accuracy: {best_threshold['accuracy']:.3f})")

# Threshold vs Accuracy grafiği
plt.figure(figsize=(10, 6))
threshs = [r['threshold'] for r in threshold_results]
accs = [r['accuracy'] for r in threshold_results]

plt.plot(threshs, accs, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Threshold vs Accuracy')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_threshold['threshold'], color='red', linestyle='--', 
            label=f'Best: {best_threshold["threshold"]:.2f}')
plt.legend()

# Threshold grafiğini kaydet
thresh_path = os.path.join(base_dir, "threshold_analysis.png")
plt.savefig(thresh_path, dpi=300, bbox_inches='tight')
print(f"Threshold analizi kaydedildi: {thresh_path}")

plt.show()

# Detaylı classification report
print(f"\nDetaylı Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Different Person', 'Same Person']))

print("\nAnalysis completed!")
