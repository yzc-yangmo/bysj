import os
import sys
from PyQt5.QtWidgets import QApplication
from gui import FoodRecognitionSystem

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    app = QApplication(sys.argv)
    window = FoodRecognitionSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()