#!/usr/bin/env python3
"""
对比原图和对齐后的图片
"""
import cv2
import numpy as np
import os

# 读取一张原图和对齐后的图
original = cv2.imread('Everyday/WIN_20250926_14_39_27_Pro.jpg')
aligned = cv2.imread('test_fixed_scale/WIN_20250926_14_39_27_Pro.jpg')

print(f"原图尺寸: {original.shape}")
print(f"对齐后尺寸: {aligned.shape}")

# 计算眼距
def get_eye_distance(img_path):
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        # 左眼中心
        left_eye = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)], axis=0)
        # 右眼中心
        right_eye = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)], axis=0)
        
        distance = np.linalg.norm(right_eye - left_eye)
        return distance, left_eye, right_eye
    return None, None, None

print("\n分析原图...")
dist1, le1, re1 = get_eye_distance('Everyday/WIN_20250926_14_39_27_Pro.jpg')
if dist1:
    print(f"原图眼距: {dist1:.1f} 像素")
    print(f"原图左眼位置: {le1}")
    print(f"原图右眼位置: {re1}")

print("\n分析对齐后的图...")
dist2, le2, re2 = get_eye_distance('test_fixed_scale/WIN_20250926_14_39_27_Pro.jpg')
if dist2:
    print(f"对齐后眼距: {dist2:.1f} 像素")
    print(f"对齐后左眼位置: {le2}")
    print(f"对齐后右眼位置: {re2}")

# 创建对比图
if original.shape == aligned.shape:
    # 并排显示
    comparison = np.hstack([original, aligned])
    
    # 缩小以便查看
    scale = 0.5
    h, w = comparison.shape[:2]
    comparison_small = cv2.resize(comparison, (int(w*scale), int(h*scale)))
    
    cv2.imwrite('comparison.jpg', comparison_small)
    print(f"\n对比图已保存到 comparison.jpg")
    print("左边是原图，右边是对齐后的图")
