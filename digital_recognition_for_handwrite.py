import sys

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import (QPainter, QPen, QImage, QFont)
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                             QHBoxLayout)

from predict import plot_predictions, load_and_preprocess_image


class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.last_point = QPoint()
        self.drawing = False
        self.setMouseTracking(True)
        self.initImage()

    def initImage(self):
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(Qt.black, 32, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def resizeEvent(self, event):
        if self.image.size() != self.size():
            self.initImage()
        super().resizeEvent(event)


class HandwritingPad(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 1000, 700)
        self.setWindowTitle('Handwrite pad')
        self.setStyleSheet("background-color: #f5f5f5;")

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(30)

        # 左侧画布区域
        canvas_container = QWidget()
        canvas_container.setStyleSheet("""
            background-color: #ffffff;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        """)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(15, 15, 15, 15)

        self.drawing_area = DrawingWidget()
        self.drawing_area.setStyleSheet("background-color: white; border: 1px solid #d0d0d0;")
        canvas_layout.addWidget(self.drawing_area)

        # 右侧控制面板
        control_panel = self.createControlPanel()

        main_layout.addWidget(canvas_container, stretch=4)
        main_layout.addWidget(control_panel)
        self.setLayout(main_layout)

    def createControlPanel(self):
        control_panel = QWidget()
        control_panel.setFixedWidth(220)
        control_panel.setStyleSheet("""
            background-color: #ffffff;
            border-radius: 8px;
            border: 2px solid #e0e0d0;
            padding: 15px;
        """)

        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(20)

        button_style = """
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #5ca2ff; }
            QPushButton:pressed { background-color: #3d7bc2; }
        """

        self.clear_btn = QPushButton("Clear")
        self.predict_btn = QPushButton("Predict")
        self.clear_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.clear_btn.setStyleSheet(button_style)
        self.predict_btn.setStyleSheet(button_style)

        control_layout.addStretch()
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.predict_btn)
        control_layout.addStretch()

        self.clear_btn.clicked.connect(self.drawing_area.clear)
        self.predict_btn.clicked.connect(self.predictNumber)

        return control_panel

    def predictNumber(self):
        # 将QImage转换为PIL Image
        qimage = self.drawing_area.image
        buffer = qimage.bits().asstring(qimage.byteCount())
        pil_image = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "RGBA", 0, 1)

        pil_image = pil_image.convert('L')
        pil_image = pil_image.resize((28, 28))
        pil_image = Image.eval(pil_image, lambda x: 255 - x)
        image_array = np.array(pil_image)
        image_array = image_array.astype('float32') / 255
        image = image_array.reshape(1, 28, 28, 1)

        plot_predictions(image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandwritingPad()
    window.show()
    print("Server running...")
    sys.exit(app.exec_())
