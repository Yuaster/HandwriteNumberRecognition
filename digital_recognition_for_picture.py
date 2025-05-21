import sys

from PyQt5.QtWidgets import QWidget, QApplication


class PictureDigitalRecognitionPad(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(800, 300, 800, 800)
        self.setWindowTitle("Picture Digital Recognition")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PictureDigitalRecognitionPad()
    window.show()
    sys.exit(app.exec_())