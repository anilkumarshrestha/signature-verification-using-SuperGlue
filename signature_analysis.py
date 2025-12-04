"""
Common Signature Analysis Module
This module is used by both improved_rotation_strategy.py and signature_matching_app.py
"""

import cv2
import torch
import numpy as np
import time
from image_preprocessing import advanced_signature_preprocessing

def rotate_image(image, angle):
    """Rotate image by given angle"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def calculate_security_score(results_by_angle, final_result):
    """Calculate advanced security score to detect potential forgeries"""
    
    # Base metrics
    ratio = final_result['ratio']
    valid_matches = final_result['valid']
    total_kpts = final_result['total']
    
    # 1. Quality over quantity check
    if valid_matches > 0:
        quality_score = min(1.0, valid_matches / max(total_kpts * 0.1, 1))  # Ideal: 10% strong matches
    else:
        quality_score = 0.0
    
    # 2. Suspicious pattern detection (rebalanced for better accuracy)
    suspicious_penalty = 0.0
    
    # Too many matches might indicate forgery attempt (more lenient)
    if ratio > 0.25 and valid_matches > 50:  # Increased thresholds
        suspicious_penalty += 0.1  # Reduced penalty
    
    # Multiple angles with similar high ratios = suspicious (more lenient)
    high_ratio_angles = [r for r in results_by_angle if r['ratio'] > 0.12]  # Higher threshold
    if len(high_ratio_angles) > 5:  # More angles needed to be suspicious
        suspicious_penalty += 0.08  # Reduced penalty
    
    # Very high ratio with low keypoints = suspicious (more targeted)
    if ratio > 0.35 and total_kpts < 10:  # Much higher ratio threshold, lower keypoint threshold
        suspicious_penalty += 0.15  # Reduced penalty
    
    # 3. Calculate final security score
    security_score = quality_score * (1.0 - suspicious_penalty)
    security_score = max(0.0, min(1.0, security_score))  # Clamp between 0-1
    
    return {
        'security_score': security_score,
        'quality_score': quality_score,
        'suspicious_penalty': suspicious_penalty,
        'risk_level': 'HIGH' if suspicious_penalty > 0.2 else 'MEDIUM' if suspicious_penalty > 0.05 else 'LOW'
    }

def analyze_signatures_with_rotation(im1, im2, matching, device, 
                                   base_threshold=0.25, 
                                   rotation_threshold=0.45,
                                   rotation_improvement_threshold=0.08,
                                   use_rotation=True,
                                   use_preprocessing=True,
                                   preprocessing_method='kmeans'):
    """
    Main signature analysis function - used in both CLI and Streamlit

    Args:
        im1, im2: Grayscale images
        matching: SuperGlue model
        device: 'cuda' or 'cpu'
        base_threshold: Base threshold value
        rotation_threshold: Rotation threshold value
        rotation_improvement_threshold: Rotation improvement threshold
        use_rotation: Whether to use rotation
        use_preprocessing: Whether to apply preprocessing (K-means)
        preprocessing_method: Preprocessing method

    Returns:
        Detailed analysis results
    """
    start_time = time.time()
    
    # IMAGE PREPROCESSING - Clean dotted paper background with K-means
    if use_preprocessing:
        print(f"Applying K-means preprocessing...")
        im1_processed = advanced_signature_preprocessing(im1, method=preprocessing_method)
        im2_processed = advanced_signature_preprocessing(im2, method=preprocessing_method)
    else:
        im1_processed = im1.copy()
        im2_processed = im2.copy()
    
    # ROTATION STRATEGY
    if use_rotation:
        rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    else:
        rotation_angles = [0]
    
    results_by_angle = []
    
    for angle in rotation_angles:
        # Rotate second image (processed version)
        im2_rotated = rotate_image(im2_processed, angle) if angle != 0 else im2_processed
        
        # Convert to tensor
        inp = {
            'image0': torch.from_numpy(im1_processed/255.).float()[None,None].to(device),
            'image1': torch.from_numpy(im2_rotated/255.).float()[None,None].to(device),
        }
        
        with torch.no_grad():
            pred = matching(inp)

        # Keypoints & matches
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = int((matches > -1).sum())
        total = max(len(kpts0), len(kpts1), 1)
        ratio = valid / total
        
        results_by_angle.append({
            'angle': angle,
            'ratio': ratio,
            'valid': valid,
            'total': total,
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches': matches,
            'im2_rotated': im2_rotated  # Processed ve rotated version
        })
    
    # IMPROVED SELECTION LOGIC
    base_result = results_by_angle[0]  # 0 degree result
    best_result = max(results_by_angle, key=lambda x: x['ratio'])
    
    # DECISION LOGIC
    final_result = base_result
    decision_threshold = base_threshold
    rotation_used = False
    
    if best_result['angle'] != 0 and use_rotation:  # Rotation found better match
        improvement = best_result['ratio'] - base_result['ratio']
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
    
    # Use final result for analysis
    result = final_result
    ratio = result['ratio']
    valid = result['valid']
    total = result['total']
    kpts0 = result['kpts0']
    kpts1 = result['kpts1']
    matches = result['matches']
    im2_rotated = result['im2_rotated']
    
    # ADVANCED SECURITY ANALYSIS
    security_analysis = calculate_security_score(results_by_angle, final_result)
    
    # Dynamic threshold adjustment based on security score
    dynamic_threshold = decision_threshold
    
    # If high risk detected, increase threshold
    if security_analysis['risk_level'] == 'HIGH':
        dynamic_threshold += 0.08  # Reduced from 0.15
    elif security_analysis['risk_level'] == 'MEDIUM':
        dynamic_threshold += 0.04  # Reduced from 0.08
    
    # Additional validation with security considerations
    if total < 20:
        dynamic_threshold += 0.03  # Further reduced from 0.05
    if ratio > 0.8 and valid < 10:
        dynamic_threshold += 0.05  # Reduced from 0.1
    
    # Make prediction with dynamic threshold
    predicted_same = ratio >= dynamic_threshold
    
    # Processing time
    processing_time = time.time() - start_time
    
    return {
        'ratio': ratio,
        'valid_matches': valid,
        'total_keypoints': total,
        'rotation_angle': result['angle'],
        'rotation_used': rotation_used,
        'threshold': dynamic_threshold,
        'base_threshold': decision_threshold,
        'predicted_same': predicted_same,
        'processing_time': processing_time,
        'all_results': results_by_angle,
        'security_analysis': security_analysis,
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'matches': matches,
        'final_image1': im1_processed,  # Processed first image
        'final_image2': im2_rotated,    # Processed and rotated second image
        'original_image1': im1,         # Original images for reference
        'original_image2': im2,
        'preprocessing_used': use_preprocessing
    }

def create_visualization(result):
    """
    Create visualization from analysis result
    """
    kpts0 = result['keypoints0']
    kpts1 = result['keypoints1']
    matches = result['matches']
    im1_processed = result['final_image1']
    im2_processed = result['final_image2']
    
    # Prepare match lines with DMatch list
    dms = [
        cv2.DMatch(_queryIdx=i, _trainIdx=int(m), _distance=0)
        for i, m in enumerate(matches) if m > -1
    ]

    # Convert keypoints to cv2.KeyPoint format
    kp0 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts0]
    kp1 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts1]

    # Convert images to BGR and draw matches (processed versions)
    im1c = cv2.cvtColor(im1_processed, cv2.COLOR_GRAY2BGR)
    im2c = cv2.cvtColor(im2_processed, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawMatches(im1c, kp0, im2c, kp1, dms, None)
    
    return vis

def add_text_overlay(vis, result, ground_truth_same=None):
    """
    Add text information to visualization
    """
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
    ratio_percent = result['ratio'] * 100
    improvement_text = ""
    if result['rotation_used']:
        base_ratio = result['all_results'][0]['ratio']
        improvement = (result['ratio'] - base_ratio) * 100
        improvement_text = f"(+{improvement:.1f}%)"
    
    info_text = f"Match: {ratio_percent:.1f}% ({result['valid_matches']}/{result['total_keypoints']}) [Rot: {result['rotation_angle']}deg] {improvement_text}"
    draw_text_with_outline(vis, info_text, (10, start_y), font, font_scale, (0, 255, 255), thickness)
    
    # Security info
    security_text = f"Security: {result['security_analysis']['risk_level']} (Score: {result['security_analysis']['security_score']:.2f})"
    draw_text_with_outline(vis, security_text, (10, start_y + line_spacing), font, font_scale, (255, 200, 0), thickness)
    
    # Threshold info
    threshold_text = f"Threshold: {result['threshold']:.2f} {'(Rot+Sec+K-means)' if result['rotation_used'] else '(Base+Sec+K-means)'}"
    draw_text_with_outline(vis, threshold_text, (10, start_y + 2*line_spacing), font, font_scale, (255, 255, 0), thickness)
    
    # Prediction
    pred_text = f"Prediction: {'SAME' if result['predicted_same'] else 'DIFFERENT'}"
    pred_color = (0, 255, 0) if result['predicted_same'] else (0, 0, 255)
    draw_text_with_outline(vis, pred_text, (10, start_y + 3*line_spacing), font, font_scale, pred_color, thickness)
    
    # Ground truth comparison (if provided)
    if ground_truth_same is not None:
        gt_text = f"Ground Truth: {'SAME' if ground_truth_same else 'DIFFERENT'}"
        correct_prediction = ground_truth_same == result['predicted_same']
        gt_color = (0, 255, 0) if correct_prediction else (0, 0, 255)
        
        draw_text_with_outline(vis, gt_text, (10, start_y + 4*line_spacing), font, font_scale, (255, 255, 255), thickness)
        
        accuracy_text = f"Result: {'CORRECT' if correct_prediction else 'WRONG'}"
        draw_text_with_outline(vis, accuracy_text, (10, start_y + 5*line_spacing), font, font_scale, gt_color, thickness)
    
    # Preprocessing info
    if result['preprocessing_used']:
        preprocess_text = "Preprocessing: K-means Applied"
        draw_text_with_outline(vis, preprocess_text, (10, start_y + 6*line_spacing), font, font_scale, (0, 255, 128), thickness)
    
    return vis

# Model loading function
def load_superglue_model():
    """
    Load SuperGlue model - singleton pattern
    """
    from models.matching import Matching
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'superpoint': dict(nms_radius=4, keypoint_threshold=0.005, max_keypoints=1024),
        'superglue': dict(weights='indoor', sinkhorn_iterations=20, match_threshold=0.2)
    }
    matching = Matching(config).to(device).eval()
    return matching, device
