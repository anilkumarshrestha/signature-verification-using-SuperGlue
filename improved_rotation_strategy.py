# Better rotation strategy to reduce false positives

import os
import cv2
import torch
import numpy as np
import json
import time
from signature_analysis import analyze_signatures_with_rotation, create_visualization, add_text_overlay, load_superglue_model

# 1) Model and device
matching, device = load_superglue_model()

# 2) Folder paths
base_dir  = r"C:\\Users\\gulme\\OneDrive\\Desktop\\dataset"
orig_root = os.path.join(base_dir, "original")
proc_root = os.path.join(base_dir, "processed")
viz_root  = os.path.join(base_dir, "visualizations_smart_rotation2") 

# Ensure directories exist
os.makedirs(base_dir, exist_ok=True)
os.makedirs(viz_root, exist_ok=True)

base_threshold = 0.25  # Lowered for better sensitivity (updated from Streamlit version)
rotation_threshold = 0.45  # Balanced threshold for rotated matches (updated from Streamlit version)
rotation_improvement_threshold = 0.08  # Higher threshold for more quality control (updated from Streamlit version)
results = []

def rotate_image(image, angle):
    """Rotate image by given angle"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def match_and_draw(im1, im2, save_path, orig_folder, proc_folder, base_threshold):
    """
    Ortak analiz fonksiyonu kullanarak imzaları karşılaştır ve görselleştir
    """
    start_time = time.time()
    
    # Ortak analiz fonksiyonunu kullan
    result = analyze_signatures_with_rotation(
        im1, im2, matching, device,
        base_threshold=base_threshold,
        rotation_threshold=0.45,
        rotation_improvement_threshold=0.08,
        use_rotation=True,
        use_preprocessing=True,
        preprocessing_method='kmeans'
    )
    
    # Debug için ön işleme görselleştirmesi
    debug_dir = os.path.join(os.path.dirname(save_path), "preprocessing_debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    debug_path1 = os.path.join(debug_dir, f"debug_{orig_folder}_{os.path.basename(save_path)}")
    debug_path2 = os.path.join(debug_dir, f"debug_{proc_folder}_{os.path.basename(save_path)}")
    
    # Ön işleme sonuçlarını kaydet (debug için)
    cv2.imwrite(debug_path1.replace('.png', '_original.png'), result['original_image1'])
    cv2.imwrite(debug_path1.replace('.png', '_cleaned.png'), result['final_image1'])
    cv2.imwrite(debug_path2.replace('.png', '_original.png'), result['original_image2'])
    cv2.imwrite(debug_path2.replace('.png', '_cleaned.png'), result['final_image2'])
    
    # Görselleştirme oluştur
    vis = create_visualization(result)
    
    # Ground truth bilgisi
    is_same_person = orig_folder == proc_folder
    
    # Metin overlay ekle
    vis_with_text = add_text_overlay(vis, result, ground_truth_same=is_same_person)
    
    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_with_text)

    return (result['ratio'], result['valid_matches'], result['total_keypoints'], 
            result['rotation_angle'], result['processing_time'], result['threshold'], 
            result['rotation_used'], result['security_analysis'])

# 3) Get only numeric folders
orig_folders = [
    d for d in os.listdir(orig_root)
    if os.path.isdir(os.path.join(orig_root, d)) and d.isdigit()
]
proc_folders = [
    d for d in os.listdir(proc_root)
    if os.path.isdir(os.path.join(proc_root, d)) and d.isdigit()
]

print("Starting SMART rotation-augmented signature matching with K-MEANS preprocessing...")
print(f"Base threshold: {base_threshold}")
print(f"Rotation threshold: {rotation_threshold}")
print(f"Rotation improvement threshold: {rotation_improvement_threshold}")
print(f"Preprocessing: K-means clustering (noktalı kağıt temizleme)")
print(f"Strategy: Use rotation only if it improves significantly, with stricter threshold")

# Track total processing time
total_start_time = time.time()
total_comparisons = 0

# 4) Process all combinations
for fo in sorted(orig_folders, key=int):
    for fp in sorted(proc_folders, key=int):
        dir_o   = os.path.join(orig_root, fo)
        dir_p   = os.path.join(proc_root, fp)
        viz_dir = os.path.join(viz_root, f"{fo}_{fp}")

        for fn_o in sorted(os.listdir(dir_o)):
            for fn_p in sorted(os.listdir(dir_p)):
                p1 = os.path.join(dir_o, fn_o)
                p2 = os.path.join(dir_p, fn_p)
                if not (os.path.isfile(p1) and os.path.isfile(p2)):
                    continue

                # Load images
                im1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
                im2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
                if im1.shape != im2.shape:
                    im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

                # Match, draw and save with smart rotation
                viz_fname = f"{os.path.splitext(fn_o)[0]}__{os.path.splitext(fn_p)[0]}_smart_rotation.png"
                viz_path  = os.path.join(viz_dir, viz_fname)
                ratio, valid, total, best_angle, processing_time, final_threshold, rotation_used, security_analysis = match_and_draw(im1, im2, viz_path, fo, fp, base_threshold)
                
                total_comparisons += 1

                # Result with smart rotation info
                results.append({
                    "orig_folder":      fo,
                    "proc_folder":      fp,
                    "orig_filename":    fn_o,
                    "proc_filename":    fn_p,
                    "match_ratio":      round(ratio, 3),
                    "matches":          valid,
                    "keypoints_total":  total,
                    "best_rotation_angle": best_angle,
                    "rotation_used":    rotation_used,
                    "final_threshold":  round(final_threshold, 3),
                    "processing_time":  round(processing_time, 3),
                    "security_score":   round(security_analysis['security_score'], 3),
                    "risk_level":       security_analysis['risk_level'],
                    "ground_truth_same": fo == fp,
                    "predicted_same":   ratio >= final_threshold,
                    "viz_path":         os.path.relpath(viz_path, base_dir)
                })
                
                rot_indicator = "✓" if rotation_used else "○"
                sec_indicator = f"[{security_analysis['risk_level']}]"
                print(f"[{fo}/{fn_o}] vs [{fp}/{fn_p}] r={ratio:.3f} t={final_threshold:.2f} {rot_indicator} {sec_indicator} (rot: {best_angle}deg)")

# Save to JSON
out_json = os.path.join(base_dir, "results_smart_rotation2.json") 
print(f"Saving results to: {out_json}")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Results successfully saved to: {out_json}")

# Calculate statistics
total_processing_time = time.time() - total_start_time
rotation_used_count = sum(1 for r in results if r['rotation_used'])

print(f"\nSmart Rotation Analysis completed!")
print(f"Results saved to: {out_json}")
print(f"Visualizations saved to: {viz_root}")
print(f"\nStrategy Statistics:")
print(f"Total comparisons: {total_comparisons}")
print(f"Rotation used: {rotation_used_count}/{total_comparisons} ({rotation_used_count/total_comparisons*100:.1f}%)")
print(f"Base threshold kept: {total_comparisons - rotation_used_count}/{total_comparisons} ({(total_comparisons - rotation_used_count)/total_comparisons*100:.1f}%)")
print(f"Average processing time: {total_processing_time/total_comparisons:.2f}s per comparison")

# Performance preview
y_true = [r["ground_truth_same"] for r in results]
y_pred = [r["predicted_same"] for r in results]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\nQuick Performance Preview:")
print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

print(f"\nSmart rotation strategy completed!")
