import cv2 as cv
import pickle
import numpy as np
import os
from ultralytics import YOLO

class PlateDetector:

    # def detect_plate_location_dynamic(self, car_roi):
    #     if car_roi is None or car_roi.size == 0:
    #         return None
            
    #     h_car, w_car = car_roi.shape[:2]
    #     car_area = w_car * h_car
        
    #     # kernel_w = max(3, int(w_car * 0.06))

    #     # أخر نتيجى توصلنالها
    #     kernel_w = max(3, int(w_car * 0.035))
    #     kernel_h = max(2, int(h_car * 0.02))

    #     # kernel_w = min(40, max(5, int(w_car * 0.045))) 
    #     # kernel_h = max(2, int(h_car * 0.02))
        
    #     min_plate_area = car_area * 0.005
    #     max_plate_area = car_area * 0.15

    #     gray = cv.cvtColor(car_roi, cv.COLOR_BGR2GRAY)

    #     gray_smooth = cv.GaussianBlur(gray, (5, 5), 0)
        
    #     # tophat_w = max(5, int(w_car * 0.08)) 
    #     # tophat_h = max(3, int(h_car * 0.04))
        
    #     # tophat_kernel = cv.getStructuringElement(cv.MORPH_RECT, (tophat_w, tophat_h))
    #     tophat_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    #     tophat = cv.morphologyEx(gray_smooth, cv.MORPH_TOPHAT, tophat_kernel)
    #     blurred = cv.GaussianBlur(tophat, (5, 5), 0)
    #     sobel_x = cv.Sobel(blurred, cv.CV_8U, dx=1, dy=0, ksize=3)
        
    #     _, threshed = cv.threshold(sobel_x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
    #     noise_kill_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    #     pruned_threshed = cv.morphologyEx(threshed, cv.MORPH_OPEN, noise_kill_kernel)

    #     dynamic_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_w, kernel_h))
    #     # closed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, dynamic_kernel)
    #     closed = cv.morphologyEx(pruned_threshed, cv.MORPH_CLOSE, dynamic_kernel)

    #     clean_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    #     closed = cv.morphologyEx(closed, cv.MORPH_OPEN, clean_kernel)
    #     #غيرت ال closed ساويتو متل ال image_debugger
    #     contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
    #     roi_center_x = w_car / 2.0
        
    #     candidates = []

    #     for contour in contours:
    #         x, y, w, h = cv.boundingRect(contour)
    #         box_area = w * h
            
    #         if h == 0 or box_area == 0:
    #             continue
            
    #         if box_area < min_plate_area or box_area > max_plate_area:
    #             continue
                
    #         aspect_ratio = float(w) / h
    #         if aspect_ratio < 1.2 or aspect_ratio > 6.0:
    #             continue
                
    #         actual_contour_area = cv.contourArea(contour)
    #         extent = float(actual_contour_area) / box_area
            
    #         if extent <= 0.45:
    #             continue

    #         box_center_x = x + (w / 2.0)
    #         distance_from_center_x = abs(roi_center_x - box_center_x)
            
    #         s_hx = max(0, 1.0 - (distance_from_center_x / roi_center_x)) 
            
    #         box_center_y = y + (h / 2.0)
    #         s_vy = box_center_y / h_car 

    #         edge_roi = threshed[y:y+h, x:x+w]
    #         white_pixels = cv.countNonZero(edge_roi)
    #         density_ratio = white_pixels / box_area
    #         s_density = min(1.0, density_ratio * 2.5) 

    #         score = box_area * extent * s_density * (s_hx * s_vy)
            
    #         candidates.append({
    #             'box': (x, y, w, h),
    #             'score': score
    #         })
                
    #     if not candidates:
    #         return None
            
    #     best_candidate = sorted(candidates, key=lambda k: k['score'], reverse=True)[0]
    #     return best_candidate['box']

    def _calculate_projection_profile_transitions(self, edge_roi: np.ndarray, w: int, h: int) -> int:
        """
        يحسب عدد الانتقالات بناءً على الإسقاط العمودي لشريحة وسطية (متين ضد الانزياح).
        """
        strip_y1, strip_y2 = int(h * 0.33), int(h * 0.66)
        if strip_y2 <= strip_y1:
            return 0 # حماية ضد الصناديق الصغيرة جداً
            
        mid_strip = edge_roi[strip_y1:strip_y2, :]
        strip_h = strip_y2 - strip_y1
        
        # جمع البكسلات البيضاء عمودياً لكل عمود في الشريحة
        col_sums = np.sum(mid_strip > 0, axis=0)
        
        # يعتبر العمود جزءاً من حرف إذا كان 20% على الأقل من بكسلاته بيضاء
        col_binary = (col_sums > (strip_h * 0.2)).astype(np.uint8)
        
        # حساب عدد الانتقالات
        transitions = np.sum(col_binary[:-1] != col_binary[1:])
        return transitions
    
    def detect_plate_location_dynamic_guassanian(self, car_roi: np.ndarray) -> tuple | None:
        """
        الوظيفة: اكتشاف موقع اللوحة في الوقت الفعلي (Video Stream) بمعمارية نظيفة.
        تتضمن بوابات رفض صارمة، دوال تقييم غاوسية، وتتبع تشخيصي للأخطاء.
        """
        if car_roi is None or car_roi.size == 0:
            return None
            
        h_car, w_car = car_roi.shape[:2]
        car_area = w_car * h_car
        roi_center_x = w_car / 2.0
        
        # تعريف أحجام الكيرنل ديناميكياً
        kernel_w = max(3, int(w_car * 0.035))
        kernel_h = max(2, int(h_car * 0.02))
        
        min_plate_area = car_area * 0.005
        max_plate_area = car_area * 0.15

        gray = cv.cvtColor(car_roi, cv.COLOR_BGR2GRAY)

        # ---------------------------------------------------------
        # Pipeline A: Adaptive Thresholding (مسار استخراج الكونتورات)
        # ---------------------------------------------------------
        blurred_a = cv.GaussianBlur(gray, (3, 3), 0)
        char_binary = cv.adaptiveThreshold(
            blurred_a, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY_INV, 15, 7
        )

        kernel_merge = cv.getStructuringElement(cv.MORPH_RECT, (kernel_w + 5, 2))
        merged_strip = cv.morphologyEx(char_binary, cv.MORPH_CLOSE, kernel_merge)

        kernel_clean = cv.getStructuringElement(cv.MORPH_RECT, (1, max(3, kernel_h // 2)))
        closed_contours_mask = cv.morphologyEx(merged_strip, cv.MORPH_OPEN, kernel_clean)

        # ---------------------------------------------------------
        # Pipeline B: Texture Extraction (مسار استخراج الملمس للتقييم)
        # ---------------------------------------------------------
        gray_smooth_b = cv.GaussianBlur(gray, (5, 5), 0)
        tophat_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
        tophat = cv.morphologyEx(gray_smooth_b, cv.MORPH_TOPHAT, tophat_kernel)
        
        blurred_b = cv.GaussianBlur(tophat, (5, 5), 0)
        sobel_x = cv.Sobel(blurred_b, cv.CV_16S, dx=1, dy=0, ksize=3)
        abs_sobel_x = cv.convertScaleAbs(sobel_x)
        
        _, threshed = cv.threshold(abs_sobel_x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # ---------------------------------------------------------
        # Logic Gates & Feature Extraction
        # ---------------------------------------------------------
        contours, _ = cv.findContours(closed_contours_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            box_area = w * h
            
            # 1. Hard Geometry Gates (بوابات الرفض الهندسية)
            if h < 3 or w < 3 or box_area == 0:
                continue
            if box_area < min_plate_area or box_area > max_plate_area:
                continue
                
            actual_contour_area = cv.contourArea(contour)
            extent = float(actual_contour_area) / box_area
            aspect_ratio = float(w) / h
            
            if aspect_ratio < 1.2 or aspect_ratio > 6.5:
                continue
            if extent < 0.35:
                continue

            # 2. Hard Density Gates (بوابات الرفض للكثافة)
            edge_roi = threshed[y:y+h, x:x+w]
            white_pixels = cv.countNonZero(edge_roi)
            density_ratio = white_pixels / box_area
            
            # قتل الصناديق الناعمة تماماً كالأسفلت وظلال السيارات
            if white_pixels < 15 or density_ratio < 0.08:
                continue 

            # ---------------------------------------------------------
            # Complex Scoring Functions (دوال التقييم الرياضية المتقدمة)
            # ---------------------------------------------------------
            
            # أ. الميزات الهندسية (Shape)
            ar_score = np.exp(-((aspect_ratio - 3.5) ** 2) / 2.0)
            s_geo = ar_score * extent  

            # ب. الموقع المكاني (Spatial X & Y)
            box_center_x = x + (w / 2.0)
            distance_from_center_x = abs(roi_center_x - box_center_x)
            s_hx = max(0, 1.0 - (distance_from_center_x / roi_center_x)) 
            
            box_center_y = y + (h / 2.0)
            y_norm = box_center_y / h_car 
            s_vy = np.exp(-((y_norm - 0.65) ** 2) / 0.15) # دالة ذروة واسعة النطاق

            # ج. كثافة الملمس (Texture Density)
            s_density = np.exp(-((density_ratio - 0.30) ** 2) / 0.04) # دالة ذروة تستهدف 30%

            # د. الانتظام النصي (Text Pattern Consistency)
            transitions = self._calculate_projection_profile_transitions(edge_roi, w, h)
            
            if 8 <= transitions <= 20:
                pattern_score = 1.0
            elif 5 <= transitions < 8 or 20 < transitions <= 28:
                pattern_score = 0.5
            else:
                pattern_score = 0.01 # أرضية منخفضة لسحق المربعات الخاطئة (كالشعارات)

            # هـ. الدمج الجدائي (Multiplicative Fusion)
            final_score = (s_geo * 1.5) * (s_hx * s_vy) * s_density * pattern_score
            
            # ---------------------------------------------------------
            # Diagnostic Telemetry (نظام التشخيص والمراقبة)
            # ---------------------------------------------------------
            print(f"[CANDIDATE BOX] x:{x}, y:{y}, w:{w}, h:{h}")
            print(f" ├─ Shape (s_geo)    : {s_geo:.3f} | ar:{aspect_ratio:.2f}, extent:{extent:.2f}")
            print(f" ├─ X-Pos (s_hx)     : {s_hx:.3f} | dist_from_center:{distance_from_center_x:.1f}")
            print(f" ├─ Y-Pos (s_vy)     : {s_vy:.3f} | y_norm:{y_norm:.2f}")
            print(f" ├─ Density (s_dens) : {s_density:.3f} | raw_ratio:{density_ratio:.3f}")
            print(f" ├─ Pattern (s_pat)  : {pattern_score:.3f} | transitions:{transitions}")
            print(f" └─ FINAL SCORE      : {final_score:.4f}\n")
            
            candidates.append({
                'box': (x, y, w, h),
                'score': final_score
            })
                
        # إرجاع أفضل مرشح إن وُجد
        if not candidates:
            return None
            
        best_candidate = sorted(candidates, key=lambda k: k['score'], reverse=True)[0]
        return best_candidate['box']



    def detect_plate_location_dynamic(self, car_roi): 
        """
        Executes a dual-pipeline architecture (Adaptive for contours, Sobel for texture).
        """
        if car_roi is None or car_roi.size == 0:
            return None
            
        h_car, w_car = car_roi.shape[:2]
        car_area = w_car * h_car
        
        kernel_w = max(3, int(w_car * 0.035))
        kernel_h = max(2, int(h_car * 0.02))
        
        min_plate_area = car_area * 0.005
        max_plate_area = car_area * 0.15

        gray = cv.cvtColor(car_roi, cv.COLOR_BGR2GRAY)

        # ---------------------------------------------------------
        # Pipeline A: Adaptive Thresholding for Contours (Shape Detection)
        # ---------------------------------------------------------
        blurred_a = cv.GaussianBlur(gray, (3, 3), 0)
        char_binary = cv.adaptiveThreshold(
            blurred_a, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY_INV, 15, 7
        )

        kernel_merge = cv.getStructuringElement(cv.MORPH_RECT, (kernel_w + 5, 2))
        merged_strip = cv.morphologyEx(char_binary, cv.MORPH_CLOSE, kernel_merge)

        kernel_clean = cv.getStructuringElement(cv.MORPH_RECT, (1, max(3, kernel_h // 2)))
        closed_contours_mask = cv.morphologyEx(merged_strip, cv.MORPH_OPEN, kernel_clean)

        # ---------------------------------------------------------
        # Pipeline B: Texture Extraction for Scoring (Edge Density)
        # ---------------------------------------------------------
        gray_smooth_b = cv.GaussianBlur(gray, (5, 5), 0)
        tophat_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
        tophat = cv.morphologyEx(gray_smooth_b, cv.MORPH_TOPHAT, tophat_kernel)
        
        blurred_b = cv.GaussianBlur(tophat, (5, 5), 0)
        sobel_x = cv.Sobel(blurred_b, cv.CV_16S, dx=1, dy=0, ksize=3)
        abs_sobel_x = cv.convertScaleAbs(sobel_x)
        
        _, threshed = cv.threshold(abs_sobel_x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # ---------------------------------------------------------
        # Logic Gates: Filtering & Extraction
        # ---------------------------------------------------------
        contours, _ = cv.findContours(closed_contours_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        roi_center_x = w_car / 2.0
        candidates = []

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            box_area = w * h
            
            # Hard Filters
            if h < 3 or w < 3 or box_area == 0:
                continue
            
            if box_area < min_plate_area or box_area > max_plate_area:
                continue
                
            actual_contour_area = cv.contourArea(contour)
            extent = float(actual_contour_area) / box_area
            aspect_ratio = float(w) / h
            
            if aspect_ratio < 1.2 or aspect_ratio > 6.5:
                continue
            if extent < 0.35:
                continue

            # ---------------------------------------------------------
            # Complex Scoring Function
            # ---------------------------------------------------------
            # 1. Geometry Features
            ar_score = np.exp(-((aspect_ratio - 3.5) ** 2) / 2.0)
            s_geo = ar_score * extent  

            # 2. Spatial Features
            box_center_x = x + (w / 2.0)
            distance_from_center_x = abs(roi_center_x - box_center_x)
            s_hx = max(0, 1.0 - (distance_from_center_x / roi_center_x)) 
            
            box_center_y = y + (h / 2.0)
            s_vy = box_center_y / h_car
            

            # 3. Texture & Pattern Features (from Pipeline B)
            edge_roi = threshed[y:y+h, x:x+w]
            
            white_pixels = cv.countNonZero(edge_roi)
            density_ratio = white_pixels / box_area
            s_density = min(1.0, density_ratio * 2.5) 
            
            # Zero-Crossings / Textual Consistency Check
            mid_row = edge_roi[h // 2, :] 
            transitions = np.sum(mid_row[:-1] != mid_row[1:])
            
            pattern_score = 0.1
            if 8 <= transitions <= 18:
                pattern_score = 1.0
            elif 5 <= transitions < 8 or 18 < transitions <= 25:
                pattern_score = 0.5
            else:
                pattern_score = 0.1

            # 4. Multiplicative Fusion
            safe_density = max(0.1, s_density)
            final_score = (s_geo * 1.5) * (s_hx * s_vy) * safe_density * pattern_score
            
            candidates.append({
                'box': (x, y, w, h),
                'score': final_score
            })
                
        if not candidates:
            return None
            
        best_candidate = sorted(candidates, key=lambda k: k['score'], reverse=True)[0]
        return best_candidate['box']

