from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QApplication, QFileDialog
import pandas as pd
import sys
import subprocess
from os import path
# Load the UI file
Ui_MainWindow, _ = uic.loadUiType("untitled.ui")


class EyeMainWindow(QWidget):
    def __init__(self):
       
        super().__init__()
        # omar's code
        # self.setupUi(self)
        # self.handle_btn()
        # self.flag = True
         # mostafa's
        self.init_ui()

        # Eye-blinking widget
        self.eyeWidget = EyeWidget()
        self.layout.addWidget(self.eyeWidget)

    def init_ui(self):
        self.layout = QVBoxLayout()

        # self.input_label = QLabel('Enter a number:')
        # self.layout.addWidget(self.input_label)

        # self.input_line = QLineEdit()
        # self.layout.addWidget(self.input_line)

        self.result_label = QLabel('Result from ff.py will appear here')
        self.layout.addWidget(self.result_label)

        browse_button = QPushButton('Browse for CSV')
        browse_button.clicked.connect(self.send_data)
        self.layout.addWidget(browse_button)


        self.setLayout(self.layout)

    def send_data(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt)")
        file_dialog.selectNameFilter("Text Files (*.txt)")
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Text File', '', 'Text Files (*.txt)')

        if file_path:
            with open(file_path, 'r') as file:
                csv_content = file.read()
            try:
                # Get the number entered by the user
                # number = float(self.input_line.text())
                # data_to_send = str(number)

                # Writing data to a file that ff.py can read
                with open('communication.txt', 'w') as file:
                    file.write(csv_content)

                # Automatically run ff.py using subprocess
                subprocess.run(['python', 'ff.py'], check=True)

                # Reading the result from ff.py
                with open('communication.txt', 'r') as file:
                    result_from_ff = file.read()

                # Update the label with the result
                self.result_label.setText(f'Result from ff.py: {result_from_ff}')

                # Update the eye blinking state based on the result
                print("Before updating LR")
                print(result_from_ff)
                print(f"Value from result_from_ff: {result_from_ff[1]}")
                self.eyeWidget.updateLR(int(result_from_ff[1]))
                print("After updating LR")

            except ValueError:
                self.result_label.setText('Please enter a valid number')
                print("error in senddata 1")
            
    
class EyeWidget(QGraphicsView):
    def __init__(self):
        super().__init__()

        # Set up the scene and view
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setSceneRect(0, 0, 400, 400)

        # Create the right eye
        self.rightEye = self.createEye(30, 50, Qt.blue, Qt.black, "right-eye")

        # Create the left eye
        self.leftEye = self.createEye(300, 50, Qt.blue, Qt.black, "left-eye")

        # Initial state: right eye open, left eye closed
        self.LR = 3
        self.updateEyeState()
        

    def createEye(self, x, y, eyeColor, pupilColor, name):
        eye = QGraphicsEllipseItem(0, 0, 200, 200)
        eye.setPos(x, y)
        eye.setBrush(QBrush(eyeColor))
        eye.setTransformOriginPoint(eye.boundingRect().center())

        pupil = QGraphicsEllipseItem(0, 0, 40, 40)
        pupil.setPos(x + 80, y + 80)
        pupil.setBrush(QBrush(pupilColor))
        pupil.setTransformOriginPoint(pupil.boundingRect().center())

        pupilout = QGraphicsEllipseItem(0, 0, 280, 200)
        pupilout.setPos(x - 50, y)
        pupilout.setBrush(QBrush(QColor("#E5E5FF")))
        pupilout.setTransformOriginPoint(pupilout.boundingRect().center())

        self.scene.addItem(pupilout)
        self.scene.addItem(eye)
        self.scene.addItem(pupil)

        shut = QGraphicsEllipseItem(0, 0, 200, 200)
        shut.setPos(x, y)
        shut.setBrush(QBrush(QColor("#E5E5FF")))
        shut.setTransformOriginPoint(shut.boundingRect().center())
        shut.setVisible(False)

        self.scene.addItem(shut)

        return shut

    def updateEyeState(self):
        # Update eye state based on LR variable
        print("entering UpdateEyeState")
        if self.LR == 0 or self.LR==1:
            self.rightEye.setVisible(self.LR == 0)
            self.leftEye.setVisible(self.LR == 1)
        else:
            self.rightEye.setVisible(self.LR == 2)
            self.leftEye.setVisible(self.LR == 2)

    def updateLR(self, new_LR):
        # Update LR variable dynamically
        self.LR = new_LR
        print("LR updated to NUMBER")
        print(f"LR updated to: {self.LR}")
        self.updateEyeState()

# Create the main window class
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.handle_btn()
        self.flag = True

    def handle_btn(self):
        self.actionbrowse.triggered.connect(self.browse)

    def browse(self):
        options = QFileDialog().options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (.csv);;All Files ()", options=options)
        if file_path:
            file_name = file_path.split("/")[-1]
            signal_data = pd.read_csv(file_name)
            print(signal_data)

            pen_colors = ["r", "b", "g", (0, 0, 255), (128, 128, 128), (192, 192, 192),
                          (0, 128, 128), (128, 0, 128), (128, 128, 0), (0, 0, 128),
                          (0, 128, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]

            # Calculate the total number of columns
            num_columns = signal_data.shape[1]
            if self.flag == True:
                # Iterate over the columns and plot each signal above each other
                for ind, (col, pen_color) in enumerate(zip(signal_data.columns, pen_colors)):
                    y_values = signal_data[col] + ind * 5  # Adjust the factor as needed
                    self.graphicsView.plot(signal_data.index, y_values, pen=pen_color, width=1)

                # Update the y-range based on the number of columns
                self.graphicsView.setYRange(0, num_columns * 5)
                self.flag = False
            else:
                for ind, (col, pen_color) in enumerate(zip(signal_data.columns, pen_colors)):
                    y_values = signal_data[col] + ind * 5  # Adjust the factor as needed
                    self.graphicsView_2.plot(signal_data.index, y_values, pen=pen_color, width=1)

                # Update the y-range based on the number of columns
                self.graphicsView_2.setYRange(0, num_columns * 5)
                self.flag = True

# Create the application and show the main window


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    eyeMainWindow = EyeMainWindow()
    eyeMainWindow.show()
    sys.exit(app.exec_())