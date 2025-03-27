import os
import sys
from PyQt5.QtWidgets import QApplication




def main():
    app = QApplication(sys.argv)
    window = FoodRecognitionSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()