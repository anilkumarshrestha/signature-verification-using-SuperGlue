import os
import cv2
import torch
import numpy as np
import json
from models.matching import Matching

# 1) Model ve cihaz
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'superpoint': dict(nms_radius=4, keypoint_threshold=0.005, max_keypoints=1024),
    'superglue':    dict(weights='indoor', sinkhorn_iterations=20, match_threshold=0.2)
}
matching = Matching(config).to(device).eval()

# 2) Klasör yolları
base_dir  = r"C:\\Users\\gulme\\OneDrive\\Desktop\\dataset"
orig_root = os.path.join(base_dir, "original")
proc_root = os.path.join(base_dir, "processed")
viz_root  = os.path.join(base_dir, "visualizations_all")
os.makedirs(viz_root, exist_ok=True)

threshold = 0.30  # Optimized threshold from confusion matrix analysis
results   = []

def match_and_draw(im1, im2, save_path, orig_folder, proc_folder, threshold):
    # Tensör dönüşümü
    inp = {
        'image0': torch.from_numpy(im1/255.).float()[None,None].to(device),
        'image1': torch.from_numpy(im2/255.).float()[None,None].to(device),
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

    # Eşleşme çizgilerini DMatch listesiyle hazırla
    dms = [
        cv2.DMatch(_queryIdx=i, _trainIdx=int(m), _distance=0)
        for i, m in enumerate(matches) if m > -1
    ]

    # Keypoints'leri cv2.KeyPoint formatına çevir
    kp0 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts0]
    kp1 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts1]

    # Görselleri BGR'ye çevir ve çiz
    im1c = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    im2c = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    vis  = cv2.drawMatches(im1c, kp0, im2c, kp1, dms, None)

    # Tahmin bilgilerini görselin üzerine yaz
    is_same_person = orig_folder == proc_folder
    predicted_same = ratio >= threshold
    
    # Metin ayarları - daha büyük ve belirgin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Sol tarafta orta yerde konumlandır (y=150 civarından başla)
    start_y = 150
    line_spacing = 25
    
    # Siyah arka plan kutusu ekle (daha iyi okunabilirlik)
    overlay = vis.copy()
    cv2.rectangle(overlay, (5, start_y-20), (350, start_y+80), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    
    # Başlık bilgileri - yüzde olarak
    ratio_percent = ratio * 100
    info_text = f"Match: {ratio_percent:.1f}% ({valid}/{total})"
    cv2.putText(vis, info_text, (10, start_y), font, font_scale, (255, 255, 255), thickness)
    
    # Ground Truth vs Prediction
    gt_text = f"GT: {'SAME' if is_same_person else 'DIFFERENT'}"
    pred_text = f"Pred: {'SAME' if predicted_same else 'DIFFERENT'}"
    
    # Renk belirleme (doğru tahmin = yeşil, yanlış = kırmızı)
    correct_prediction = is_same_person == predicted_same
    color = (0, 255, 0) if correct_prediction else (0, 0, 255)  # BGR format
    
    cv2.putText(vis, gt_text, (10, start_y + line_spacing), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(vis, pred_text, (10, start_y + 2*line_spacing), font, font_scale, color, thickness)
    
    # Doğruluk durumu
    accuracy_text = f"Result: {'CORRECT' if correct_prediction else 'WRONG'}"
    cv2.putText(vis, accuracy_text, (10, start_y + 3*line_spacing), font, font_scale, color, thickness+1)

    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)

    return ratio, valid, total

# 3) Sadece sayısal klasörleri al
orig_folders = [
    d for d in os.listdir(orig_root)
    if os.path.isdir(os.path.join(orig_root, d)) and d.isdigit()
]
proc_folders = [
    d for d in os.listdir(proc_root)
    if os.path.isdir(os.path.join(proc_root, d)) and d.isdigit()
]

# 4) Tüm kombinasyonları işle
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

                # Görselleri yükle
                im1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
                im2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
                if im1.shape != im2.shape:
                    im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

                # Eşleştir & Çiz ve kaydet
                viz_fname = f"{os.path.splitext(fn_o)[0]}__{os.path.splitext(fn_p)[0]}_match.png"
                viz_path  = os.path.join(viz_dir, viz_fname)
                ratio, valid, total = match_and_draw(im1, im2, viz_path, fo, fp, threshold)

                # Sonuç
                results.append({
                    "orig_folder":      fo,
                    "proc_folder":      fp,
                    "orig_filename":    fn_o,
                    "proc_filename":    fn_p,
                    "match_ratio":      round(ratio, 3),
                    "matches":          valid,
                    "keypoints_total":  total,
                    "ground_truth_same": fo == fp,
                    "predicted_same":   ratio >= threshold,
                    "viz_path":         os.path.relpath(viz_path, base_dir)
                })
                print(f"[{fo}/{fn_o}] vs [{fp}/{fn_p}] r={ratio:.3f}")

# 5) JSON'a yaz
out_json = os.path.join(base_dir, "results2.json")  # Yeni threshold sonuçları
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("JSON and visualizations saved (Threshold 0.30).")