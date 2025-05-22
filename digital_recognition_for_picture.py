import sys
import cv2
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QFileDialog, QMessageBox)
from matplotlib import pyplot as plt

from predict import load_and_preprocess_image, plot_predictions
from yolo_about.handwrite_number_box import DigitDetector


class ImageRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.filepath = ''
        self.initUI()

    def initUI(self):
        # 设置窗口属性
        self.setGeometry(800, 300, 800, 800)
        self.setWindowTitle("Image Digit Recognition")
        self.setStyleSheet("background-color: #f5f7fa;")

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # 标题标签
        title_label = QLabel("Image Digit Recognition System")
        title_font = QFont("Arial", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            color: #2c3e50;
            margin-bottom: 25px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        """)
        main_layout.addWidget(title_label)

        # 图片预览区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(760, 400)
        self.image_label.setStyleSheet("""
            border: 2px dashed #a0a0a0;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        """)

        # 添加初始提示文本和图标
        self.image_label.setText("Upload Image\nSupported formats: JPG, PNG, BMP")
        self.image_label.setFont(QFont("Arial", 14, QFont.Light))
        main_layout.addWidget(self.image_label)

        # 上传按钮
        upload_button = QPushButton("Browse Image")
        upload_button.setFixedSize(180, 50)
        upload_button.setFont(QFont("Arial", 8, QFont.Bold))
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                border: none;
                box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
            }
            QPushButton:hover {
                background-color: #2980b9;
                box-shadow: 0 6px 8px rgba(52, 152, 219, 0.3);
            }
            QPushButton:pressed {
                background-color: #2471a3;
                box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
            }
        """)
        upload_button.clicked.connect(self.select_image)

        recognition_button = QPushButton("Recognition")
        recognition_button.setFixedSize(180, 50)
        recognition_button.setFont(QFont("Arial", 8, QFont.Bold))
        recognition_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                border: none;
                box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
            }
            QPushButton:hover {
                background-color: #2980b9;
                box-shadow: 0 6px 8px rgba(52, 152, 219, 0.3);
            }
            QPushButton:pressed {
                background-color: #2471a3;
                box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
            }
        """)
        recognition_button.clicked.connect(self.recognition)

        main_layout.addWidget(upload_button)
        main_layout.addWidget(recognition_button)

        # 状态栏
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #7f8c8d; margin-top: 15px;")
        main_layout.addWidget(self.status_label)

        # 设置布局
        self.setLayout(main_layout)

    def select_image(self):
        # 更新状态
        self.status_label.setText("Opening file dialog...")

        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )

        if file_path:
            try:
                self.status_label.setText(f"Loading image: {file_path.split('/')[-1]}")
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        600, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.image_label.setPixmap(scaled_pixmap)
                    self.image_label.setText("")
                    self.status_label.setText("Image loaded successfully")
                    self.filepath = file_path
                else:
                    self.status_label.setText("Failed to load image")
                    QMessageBox.warning(self, "Error", "Failed to load image. Please check the file format.")
            except Exception as e:
                self.status_label.setText(f"Error: {str(e)}")
                QMessageBox.warning(self, "Error", f"An error occurred while loading the image:\n{str(e)}")
        else:
            self.status_label.setText("Image selection cancelled")

    def recognition(self):
        detector = DigitDetector()
        test_source = self.filepath
        output_dir = "yolo_about/result_img_for_predict"

        result_image, digits = detector.detect_and_extract_digits(
            test_source,
            output_dir,
            use_processed=True
        )

        if result_image is not None:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Digit Extraction from Processed Image")
            plt.show()

        if digits:
            for i, digit in enumerate(digits):
                digit_pil = Image.fromarray(cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY))
                processed_image = load_and_preprocess_image(digit_pil, is_pil_image=True)
                plot_predictions(processed_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageRecognitionApp()
    window.show()
    print("Server running...")
    sys.exit(app.exec_())