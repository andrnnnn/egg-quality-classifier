import sys
from PyQt5.QtWidgets import QApplication
from modules.ui import EggQualityApp

def main():
    app = QApplication(sys.argv)
    window = EggQualityApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
