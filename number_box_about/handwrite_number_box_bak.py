import cv2
import numpy as np
import os
import base64


class DigitDetector:
    def __init__(self):
        self.clahe_clip_limit = 3.0
        self.clahe_grid_size = (8, 8)

    def _preprocess_base(self, image, debug=False):
        # 1. 颜色空间分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. 判断背景类型（亮色/暗色）
        mean_brightness = np.mean(gray)
        is_dark_bg = mean_brightness < 127

        # 3. 计算图像熵，评估对比度
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))

        # 4. 动态调整对比度
        if entropy < 4.0:
            clip_limit = min(self.clahe_clip_limit * (5.0 - entropy), 8.0)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.clahe_grid_size)
            gray = clahe.apply(gray)

        # 5. 创建颜色掩码（针对彩色背景）
        hue = hsv[:, :, 0]
        hue_variance = np.var(hue)
        color_mask = np.zeros_like(gray)

        if hue_variance > 100:  # 彩色背景
            # 尝试分离深色数字
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 70])
            color_mask = cv2.inRange(hsv, lower_black, upper_black)

        # 6. 选择合适的阈值方法
        median = cv2.medianBlur(gray, 3)
        filtered = cv2.bilateralFilter(median, 7, 50, 50)

        if is_dark_bg:
            _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 7. 融合多种方法的结果
        combined = cv2.bitwise_or(thresh, color_mask)

        # 8. 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        # 9. 形态学重建
        seed = opened.copy()
        kernel_reconstruct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for _ in range(10):
            dilated = cv2.dilate(seed, kernel_reconstruct)
            seed = cv2.bitwise_and(dilated, opened)
        reconstructed = seed.copy()

        if debug:
            try:
                debug_path = "number_box_about/result_debug/debug_reconstructed.jpg"
                cv2.imwrite(debug_path, reconstructed)
            except Exception as e:
                print(f"保存调试图像失败: {e}")

        return reconstructed

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
