# Better rotation strategy to reduce false positives

import os
import cv2
import torch
import numpy as np
import json
import time
from models.matching import Matching

# 1) Model and device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'superpoint': dict(nms_radius=4, keypoint_threshold=0.005, max_keypoints=1024),
    'superglue': dict(weights='indoor', sinkhorn_iterations=20, match_threshold=0.2)
}
matching = Matching(config).to(device).eval()

# 2) Folder paths
base_dir  = r"C:\\Users\\gulme\\OneDrive\\Desktop\\dataset"
orig_root = os.path.join(base_dir, "original")
proc_root = os.path.join(base_dir, "processed")
viz_root  = os.path.join(base_dir, "visualizations_smart_rotation2") 

# Ensure directories exist
os.makedirs(base_dir, exist_ok=True)
os.makedirs(viz_root, exist_ok=True)

base_threshold = 0.30  # Increased from 0.35 to reduce FP
rotation_threshold = 0.50  # Increased from 0.45 to reduce FP  
rotation_improvement_threshold = 0.10  # Minimum improvement needed to accept rotation
results = []

def rotate_image(image, angle):
    """Rotate image by given angle"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def match_and_draw(im1, im2, save_path, orig_folder, proc_folder, base_threshold):
    start_time = time.time()
    
    # ROTATION STRATEGY
    rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    results_by_angle = []
    
    for angle in rotation_angles:
        # Rotate second image
        im2_rotated = rotate_image(im2, angle) if angle != 0 else im2
        
        # Convert to tensor
        inp = {
            'image0': torch.from_numpy(im1/255.).float()[None,None].to(device),
            'image1': torch.from_numpy(im2_rotated/255.).float()[None,None].to(device),
        }
        with torch.no_grad():
            pred = matching(inp)

        # Keypoints & matches
        kpts0   = pred['keypoints0'][0].cpu().numpy()
        kpts1   = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = int((matches > -1).sum())
        total = max(len(kpts0), len(kpts1), 1)
        ratio = valid / total
        
        results_by_angle.append({
            'angle': angle,
            'ratio': ratio, # Her açının performansı kaydediliyor
            'valid': valid, # Eşleşen keypoint sayısı
            'total': total, # Toplam keypoint sayısı
            'kpts0': kpts0, # İlk imzanın keypoint'leri
            'kpts1': kpts1, # İkinci imzanın keypoint'leri
            'matches': matches, # Eşleştirme array'i
            'im2_rotated': im2_rotated # Döndürülmüş görüntü
        })
    
    # IMPROVED SELECTION LOGIC
    base_result = results_by_angle[0]  # 0 degree result
    best_result = max(results_by_angle, key=lambda x: x['ratio'])
    
    # DECISION LOGIC:
    # 1. If base result is good enough, use it (no rotation needed)
    # 2. If rotation improves significantly, use it but with stricter threshold
    # 3. If rotation doesn't improve much, stick with base result
    
    final_result = base_result # # En yüksek ratio
    decision_threshold = base_threshold
    rotation_used = False
    
    if best_result['angle'] != 0:  # Rotation found better match
        improvement = best_result['ratio'] - base_result['ratio']
        # En iyi sonuç 0° değilse Rotation analizi yap
        # Only use rotation if improvement is significant
        if improvement >= rotation_improvement_threshold:
            final_result = best_result
            rotation_used = True
            # Use stricter threshold for rotated matches
            decision_threshold = rotation_threshold
        else:
            # Improvement not significant enough, stick with base
            final_result = base_result
            decision_threshold = base_threshold
    
    # Use final result for visualization
    result = final_result
    ratio = result['ratio']
    valid = result['valid']
    total = result['total']
    kpts0 = result['kpts0']
    kpts1 = result['kpts1']
    matches = result['matches']
    im2_rotated = result['im2_rotated']
    
    # ADDITIONAL VALIDATION: Check match consistency
    # If too few keypoints, be more conservative
    if total < 20:
        decision_threshold += 0.1
    ''''Eğer toplam keypoint sayısı 20'den azsa, threshold'u 0.1 artırıyor. 
        Az keypoint varsa güvenilirlik düşüktür'''

    # If very high ratio but few matches, be suspicious
    if ratio > 0.8 and valid < 10:
        decision_threshold += 0.1 

    '''Ratio çok yüksek (%80+) ama eşleşme sayısı az (<10) ise, 
    threshold'u artırıyor. Neden: Bu durum "şüpheli" - 
    az sayıda keypoint'ten yüksek oran şans eseri olabilir.'''

    # Make prediction
    predicted_same = ratio >= decision_threshold
    
    # Prepare match lines with DMatch list
    dms = [
        cv2.DMatch(_queryIdx=i, _trainIdx=int(m), _distance=0)
        for i, m in enumerate(matches) if m > -1
    ]

    # Convert keypoints to cv2.KeyPoint format
    kp0 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts0]
    kp1 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts1]

    # Convert images to BGR and draw matches
    im1c = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    im2c = cv2.cvtColor(im2_rotated, cv2.COLOR_GRAY2BGR)
    vis  = cv2.drawMatches(im1c, kp0, im2c, kp1, dms, None)

    # Add prediction info to image
    is_same_person = orig_folder == proc_folder
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    start_y = 25
    line_spacing = 20
    
    # Simple text with outline function
    def draw_text_with_outline(img, text, position, font, scale, color, thickness, outline_color=(0, 0, 0)):
        cv2.putText(img, text, position, font, scale, outline_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)
    
    # Header information
    processing_time = time.time() - start_time
    ratio_percent = ratio * 100
    improvement_text = f"(+{(best_result['ratio'] - base_result['ratio'])*100:.1f}%)" if rotation_used else ""
    
    info_text = f"Match: {ratio_percent:.1f}% ({valid}/{total}) [Rot: {result['angle']}deg] {improvement_text}"
    draw_text_with_outline(vis, info_text, (10, start_y), font, font_scale, (0, 255, 255), thickness)
    
    # Decision info
    decision_text = f"Threshold: {decision_threshold:.2f} {'(Rotated)' if rotation_used else '(Base)'}"
    draw_text_with_outline(vis, decision_text, (10, start_y + line_spacing), font, font_scale, (255, 255, 0), thickness)
    
    # Ground Truth vs Prediction
    gt_text = f"GT: {'SAME' if is_same_person else 'DIFFERENT'}"
    pred_text = f"Pred: {'SAME' if predicted_same else 'DIFFERENT'}"
    
    # Color determination
    correct_prediction = is_same_person == predicted_same
    color = (0, 255, 0) if correct_prediction else (0, 0, 255)
    
    draw_text_with_outline(vis, gt_text, (10, start_y + 2*line_spacing), font, font_scale, (255, 255, 255), thickness)
    draw_text_with_outline(vis, pred_text, (10, start_y + 3*line_spacing), font, font_scale, color, thickness)
    
    # Result
    accuracy_text = f"Result: {'CORRECT' if correct_prediction else 'WRONG'}"
    draw_text_with_outline(vis, accuracy_text, (10, start_y + 4*line_spacing), font, font_scale, color, thickness)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)

    return ratio, valid, total, result['angle'], processing_time, decision_threshold, rotation_used

# 3) Get only numeric folders
orig_folders = [
    d for d in os.listdir(orig_root)
    if os.path.isdir(os.path.join(orig_root, d)) and d.isdigit()
]
proc_folders = [
    d for d in os.listdir(proc_root)
    if os.path.isdir(os.path.join(proc_root, d)) and d.isdigit()
]

print("Starting SMART rotation-augmented signature matching...")
print(f"Base threshold: {base_threshold}")
print(f"Rotation threshold: {rotation_threshold}")
print(f"Rotation improvement threshold: {rotation_improvement_threshold}")
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
                ratio, valid, total, best_angle, processing_time, final_threshold, rotation_used = match_and_draw(im1, im2, viz_path, fo, fp, base_threshold)
                
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
                    "ground_truth_same": fo == fp,
                    "predicted_same":   ratio >= final_threshold,
                    "viz_path":         os.path.relpath(viz_path, base_dir)
                })
                
                rot_indicator = "✓" if rotation_used else "○"
                print(f"[{fo}/{fn_o}] vs [{fp}/{fn_p}] r={ratio:.3f} t={final_threshold:.2f} {rot_indicator} (rot: {best_angle}deg)")

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
