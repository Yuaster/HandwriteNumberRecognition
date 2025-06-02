import cv2
import numpy as np
import os
import base64


class DigitDetector:
    def __init__(self):
        self.clahe_clip_limit = 3.0
        self.clahe_grid_size = (8, 8)
        self.aspect_ratio_range = (0.9, 1.1)  # 合适的宽高比范围
        self.dark_bg_threshold = 127  # 背景亮度阈值，可调整

    def _preprocess_base(self, image, debug=False):
        # 0. 提前判断背景类型
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        is_dark_bg = mean_brightness < self.dark_bg_threshold

        if debug:
            print(f"背景亮度: {mean_brightness}, 是否深色背景: {is_dark_bg}")

        # 1. 只有在浅色背景下才进行阴影检测
        shadow_mask = None
        if not is_dark_bg:
            shadow_mask = self._detect_shadow(image, debug)

        # 2. 对原图进行常规预处理
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 动态调整对比度
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=self.clahe_grid_size)
        gray = clahe.apply(gray)

        # 创建颜色掩码
        hue = hsv[:, :, 0]
        hue_variance = np.var(hue)
        color_mask = np.zeros_like(gray)

        if hue_variance > 100:  # 彩色背景
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 70])
            color_mask = cv2.inRange(hsv, lower_black, upper_black)

        # 选择合适的阈值方法
        median = cv2.medianBlur(gray, 3)
        filtered = cv2.bilateralFilter(median, 7, 50, 50)

        if is_dark_bg:
            _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 形态学重建
        seed = opened.copy()
        kernel_reconstruct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for _ in range(5):
            dilated = cv2.dilate(seed, kernel_reconstruct)
            seed = cv2.bitwise_and(dilated, opened)
        reconstructed = seed.copy()

        # 3. 阴影区域处理（仅对浅色背景进行）
        if not is_dark_bg and shadow_mask is not None:
            # 创建阴影区域的掩码
            shadow_indices = np.where(shadow_mask > 0)

            # 在阴影区域内，只保留reconstructed中与shadow_mask重叠的部分
            shadow_overlap = np.zeros_like(reconstructed)
            shadow_overlap[shadow_indices] = reconstructed[shadow_indices]

            result = shadow_overlap
        else:
            # 深色背景直接使用常规预处理结果
            result = reconstructed

        # 4. 后处理优化
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)

        if debug:
            try:
                cv2.imwrite("number_box_about/result_debug/debug_final_result.jpg", result)
                cv2.imwrite("number_box_about/result_debug/debug_reconstructed.jpg", reconstructed)
                if shadow_mask is not None:
                    cv2.imwrite("number_box_about/result_debug/debug_shadow_mask.jpg", shadow_mask)
            except Exception as e:
                print(f"保存调试图像失败: {e}")

        return result

    def _detect_shadow(self, image, debug=False):
        """基于亮度分析的阴影检测"""
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 计算亮度通道的统计特征
        l_mean = np.mean(l)
        l_std = np.std(l)

        # 动态确定低亮度阈值
        low_threshold = max(0, l_mean - l_std * 1.2)  # 可调整参数

        # 创建低亮度掩码
        low_luminance = np.zeros_like(l)
        low_luminance[l < low_threshold] = 255

        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)

        # 自适应阈值处理
        _, adaptive_thresh = cv2.threshold(enhanced_l, 0, 255,
                                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        final_mask = low_luminance

        if debug:
            try:
                cv2.imwrite("number_box_about/result_debug/debug_low_luminance.jpg", low_luminance)
                cv2.imwrite("number_box_about/result_debug/debug_adaptive_thresh.jpg", adaptive_thresh)
                cv2.imwrite("number_box_about/result_debug/debug_shadow_mask.jpg", final_mask)
            except Exception as e:
                print(f"保存阴影检测调试图像失败: {e}")

        return final_mask

    def _preprocess_for_visual(self, image, debug=False):
        reconstructed = self._preprocess_base(image, debug)
        result = cv2.dilate(reconstructed, (2, 2), iterations=1)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def _preprocess_for_contour(self, image, debug=False):
        reconstructed = self._preprocess_base(image, debug)

        contours, _ = cv2.findContours(reconstructed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(reconstructed)
        min_area = image.shape[0] * image.shape[1] * 0.0005

        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        result = cv2.dilate(mask, (2, 2), iterations=1)
        if debug:
            try:
                debug_path = "number_box_about/result_debug/debug_result.jpg"
                cv2.imwrite(debug_path, result)
            except Exception as e:
                print(f"保存调试图像失败: {e}")
        return result

    def detect_and_extract_digits(self, image_source, output_dir=None, padding=10, use_processed=True):
        try:
            image = self._load_image(image_source)
            processed_mask = self._preprocess_for_contour(image, debug=True)

            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_image = image.copy()
            digit_images = []
            digit_boxes = []

            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)

                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)

                if use_processed:
                    processed_rgb = self._preprocess_for_visual(image, debug=True)
                    digit_roi = processed_rgb[y1:y2, x1:x2]
                else:
                    digit_roi = image[y1:y2, x1:x2]

                # 检测并调整长宽比
                digit_roi = self._ensure_aspect_ratio(digit_roi)

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    digit_path = os.path.join(output_dir, f"digit_{i}.png")
                    cv2.imwrite(digit_path, digit_roi)

                digit_images.append(digit_roi)
                digit_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return result_image, digit_images, digit_boxes

        except Exception as e:
            print(f"处理错误: {e}")
            return None, [], []

    def _load_image(self, image_source):
        if isinstance(image_source, str) and image_source.startswith("data:image"):
            return self.decode_base64_image(image_source)
        else:
            img = cv2.imread(image_source)
            if img is None:
                raise FileNotFoundError(f"无效的输入源: {image_source}")
            return img

    @staticmethod
    def decode_base64_image(image_data):
        header, data = image_data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _ensure_aspect_ratio(self, image):
        """确保图像的宽高比在合适范围内，否则填充纯黑底色"""
        if image.size == 0:
            return image

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return image

        aspect_ratio = w / h

        # 如果长宽比在合适范围内，直接返回原图
        if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
            return image

        if aspect_ratio < self.aspect_ratio_range[0]:  # 太窄，增加宽度
            target_w = int(h * self.aspect_ratio_range[0])
            padding = (target_w - w) // 2
            padded = cv2.copyMakeBorder(
                image, 0, 0, padding, target_w - w - padding,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            target_h = int(w / self.aspect_ratio_range[1])
            padding = (target_h - h) // 2
            padded = cv2.copyMakeBorder(
                image, padding, target_h - h - padding, 0, 0,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        return padded