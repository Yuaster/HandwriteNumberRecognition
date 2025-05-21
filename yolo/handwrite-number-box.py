import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import os


class DigitDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def _preprocess_base(self, image, debug=False):
        """公共预处理流程"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 3)
        filtered = cv2.bilateralFilter(median, 7, 50, 50)

        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        seed = opened.copy()
        kernel_reconstruct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for _ in range(10):
            dilated = cv2.dilate(seed, kernel_reconstruct)
            seed = cv2.bitwise_and(dilated, opened)
        reconstructed = seed.copy()

        if debug:
            cv2.imwrite("result_debug/debug_reconstructed.jpg", reconstructed)
        return reconstructed

    def _preprocess_for_visual(self, image, debug=False):
        """用于可视化的预处理（返回RGB图像）"""
        reconstructed = self._preprocess_base(image, debug)
        # 转换为RGB并强化边缘
        result = cv2.dilate(reconstructed, (2, 2), iterations=1)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def _preprocess_for_contour(self, image, debug=False):
        """用于轮廓检测的预处理"""
        reconstructed = self._preprocess_base(image, debug)

        contours, _ = cv2.findContours(reconstructed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(reconstructed)
        min_area = image.shape[0] * image.shape[1] * 0.0005

        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        result = cv2.dilate(mask, (2, 2), iterations=1)
        if debug:
            cv2.imwrite("result_debug/debug_result.jpg", result)
        return result

    def detect_and_extract_digits(self, image_source, output_dir=None, padding=10, use_processed=True):
        """检测并提取单个数字图像（支持从预处理后图像裁剪）"""
        try:
            image = self._load_image(image_source)
            processed_mask = self._preprocess_for_contour(image, debug=True)

            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_image = image.copy()
            digit_images = []

            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)

                # 添加边界填充
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)

                # 根据选项选择裁剪原图还是预处理后的图像
                if use_processed:
                    # 创建与原图相同大小的预处理后图像（RGB）
                    processed_rgb = self._preprocess_for_visual(image, debug=False)
                    digit_roi = processed_rgb[y1:y2, x1:x2]
                else:
                    digit_roi = image[y1:y2, x1:x2]

                # 保存数字图像
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    digit_path = os.path.join(output_dir, f"digit_{i}.png")
                    cv2.imwrite(digit_path, digit_roi)

                digit_images.append(digit_roi)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return result_image, digit_images

        except Exception as e:
            print(f"处理错误: {e}")
            return None, []

    def _load_image(self, image_source):
        """统一图像加载逻辑"""
        if isinstance(image_source, str) and image_source.startswith("data:image"):
            return decode_base64_image(image_source)
        else:
            img = cv2.imread(image_source)
            if img is None:
                raise FileNotFoundError(f"无效的输入源: {image_source}")
            return img


def decode_base64_image(image_data):
    header, data = image_data.split(",", 1)
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


if __name__ == "__main__":
    detector = DigitDetector()
    test_source = "test_image/img.png"
    output_dir = "result_img_for_predict"

    # 从预处理后图像裁剪
    result_image, digits = detector.detect_and_extract_digits(
        test_source,
        output_dir,
        use_processed=True  # 设为True表示从预处理后图像裁剪
    )

    if result_image is not None:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Digit Extraction from Processed Image")
        plt.show()

        # 显示提取的数字
        if digits:
            fig, axes = plt.subplots(1, len(digits), figsize=(15, 3))
            for i, digit in enumerate(digits):
                axes[i].imshow(cv2.cvtColor(digit, cv2.COLOR_BGR2RGB))
                axes[i].axis("off")
                axes[i].set_title(f"Digit {i}")
            plt.tight_layout()
            plt.show()