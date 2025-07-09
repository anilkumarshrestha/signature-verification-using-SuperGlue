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
viz_root  = os.path.join(base_dir, "visualizations_with_rotation2")

# Ensure directories exist
os.makedirs(base_dir, exist_ok=True)
os.makedirs(viz_root, exist_ok=True)
print(f"Base directory: {base_dir}")
print(f"Visualization directory: {viz_root}")

threshold = 0.45  # Increased threshold to reduce false positives
results   = []

def rotate_image(image, angle):
    """Rotate image by given angle"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def match_and_draw(im1, im2, save_path, orig_folder, proc_folder, threshold):
    start_time = time.time()
    
    # rotation testing: test 0° first, then others if needed
    rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    best_ratio = 0
    best_result = None
    angles_tested = 0
    
    for i, angle in enumerate(rotation_angles):
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
        angles_tested += 1
        
        # Keep the best match
        if ratio > best_ratio:
            best_ratio = ratio
            best_result = {
                'angle': angle,
                'kpts0': kpts0,
                'kpts1': kpts1,
                'matches': matches,
                'valid': valid,
                'total': total,
                'ratio': ratio,
                'im2_rotated': im2_rotated,
                'angles_tested': angles_tested
            }
        
        # Early stopping: if good enough match found, don't test more angles
        if ratio > 0.5:  # Increased early stopping threshold
            break
            
        # Smart stopping: if first angle (0°) is very bad, test all angles
        # if first angle is decent, only test key angles (0°, 90°, 180°, 270°)
        if i == 0 and ratio > 0.25:  # Slightly increased smart stopping threshold
            rotation_angles = [0, 90, 180, 270]  # Only test main angles
    
    # Use best result for visualization (handle None case)
    if best_result is None:
        # Fallback: create a dummy result if no match found
        best_result = {
            'angle': 0,
            'kpts0': np.array([]),
            'kpts1': np.array([]),
            'matches': np.array([]),
            'valid': 0,
            'total': 1,
            'ratio': 0.0,
            'im2_rotated': im2,
            'angles_tested': angles_tested
        }
    
    result = best_result
    kpts0 = result['kpts0']
    kpts1 = result['kpts1'] 
    matches = result['matches']
    valid = result['valid']
    total = result['total']
    ratio = result['ratio']
    im2_rotated = result['im2_rotated']

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
    predicted_same = ratio >= threshold
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    start_y = 25
    line_spacing = 20
    
    # Simple text with outline function
    def draw_text_with_outline(img, text, position, font, scale, color, thickness, outline_color=(0, 0, 0)):
        # First black outline
        cv2.putText(img, text, position, font, scale, outline_color, thickness + 1, cv2.LINE_AA)
        # Then main color
        cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)
    
    # Header information with rotation angle and processing time
    processing_time = time.time() - start_time
    ratio_percent = ratio * 100
    info_text = f"Match: {ratio_percent:.1f}% ({valid}/{total}) [Rot: {result['angle']}deg] [Tested: {result['angles_tested']}/8] [Time: {processing_time:.2f}s]"
    draw_text_with_outline(vis, info_text, (10, start_y), font, font_scale, (0, 255, 255), thickness)
    
    # Ground Truth vs Prediction
    gt_text = f"GT: {'SAME' if is_same_person else 'DIFFERENT'}"
    pred_text = f"Pred: {'SAME' if predicted_same else 'DIFFERENT'}"
    
    # Color determination
    correct_prediction = is_same_person == predicted_same
    color = (0, 255, 0) if correct_prediction else (0, 0, 255)
    
    draw_text_with_outline(vis, gt_text, (10, start_y + line_spacing), font, font_scale, (255, 255, 255), thickness)
    draw_text_with_outline(vis, pred_text, (10, start_y + 2*line_spacing), font, font_scale, color, thickness)
    
    # Accuracy status
    accuracy_text = f"Result: {'CORRECT' if correct_prediction else 'WRONG'}"
    draw_text_with_outline(vis, accuracy_text, (10, start_y + 3*line_spacing), font, font_scale, color, thickness)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)

    return ratio, valid, total, result['angle'], processing_time, result['angles_tested']  # Also return processing time and angles tested

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
print(f"Smart rotation: Early stopping at 50% match, adaptive angle testing")
print(f"Decision threshold: {threshold} (signatures with ratio >= {threshold} considered 'same')")

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

                # Match, draw and save with rotation
                viz_fname = f"{os.path.splitext(fn_o)[0]}__{os.path.splitext(fn_p)[0]}_rotation_match.png"
                viz_path  = os.path.join(viz_dir, viz_fname)
                ratio, valid, total, best_angle, processing_time, angles_tested = match_and_draw(im1, im2, viz_path, fo, fp, threshold)
                
                total_comparisons += 1

                # Result with rotation info
                results.append({
                    "orig_folder":      fo,
                    "proc_folder":      fp,
                    "orig_filename":    fn_o,
                    "proc_filename":    fn_p,
                    "match_ratio":      round(ratio, 3),
                    "matches":          valid,
                    "keypoints_total":  total,
                    "best_rotation_angle": best_angle,
                    "angles_tested": angles_tested,
                    "processing_time":  round(processing_time, 3),
                    "ground_truth_same": fo == fp,
                    "predicted_same":   ratio >= threshold,
                    "viz_path":         os.path.relpath(viz_path, base_dir)
                })
                print(f"[{fo}/{fn_o}] vs [{fp}/{fn_p}] r={ratio:.3f} (rot: {best_angle}deg) (tested: {angles_tested}/8) (time: {processing_time:.2f}s)")

# Save to JSON with rotation info
out_json = os.path.join(base_dir, "results_with_rotation2.json") 
print(f"Saving results to: {out_json}")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Results successfully saved to: {out_json}")

# Calculate total processing time
total_processing_time = time.time() - total_start_time
avg_time_per_comparison = total_processing_time / total_comparisons if total_comparisons > 0 else 0

print(f"\nRotation-augmented analysis completed!")
print(f"Results saved to: {out_json}")
print(f"Visualizations saved to: {viz_root}")
print(f"\nTiming Statistics:")
print(f"Total comparisons: {total_comparisons}")
print(f"Total processing time: {total_processing_time:.2f} seconds")
print(f"Average time per comparison: {avg_time_per_comparison:.2f} seconds")
print(f"Estimated time with 8x rotation overhead: {avg_time_per_comparison/8:.2f}s per single match")

# Calculate rotation statistics
rotation_stats = {}
time_stats = []

for result in results:
    angle = result['best_rotation_angle']
    if angle not in rotation_stats:
        rotation_stats[angle] = 0
    rotation_stats[angle] += 1
    time_stats.append(result['processing_time'])

print(f"\nRotation Statistics:")
for angle in sorted(rotation_stats.keys()):
    count = rotation_stats[angle]
    percentage = (count / len(results)) * 100
    print(f"  {angle}deg: {count} matches ({percentage:.1f}%)")

# Time statistics
if time_stats:
    min_time = min(time_stats)
    max_time = max(time_stats)
    avg_time = sum(time_stats) / len(time_stats)
    
    print(f"\nProcessing Time Statistics:")
    print(f"  Fastest match: {min_time:.2f}s")
    print(f"  Slowest match: {max_time:.2f}s")
    print(f"  Average match: {avg_time:.2f}s")
