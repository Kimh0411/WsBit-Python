from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QMovie

class Sticker(QtWidgets.QMainWindow):
    def __init__(self, img_path, xy, size=1.0, on_top=False):
        super(Sticker, self).__init__()
        self.timer = QtCore.QTimer(self)
        self.img_path = img_path
        self.xy = xy
        self.from_xy = xy
        self.from_xy_diff = [0, 0]
        self.to_xy = xy
        self.to_xy_diff = [0, 0]
        self.speed = 60
        # x: 0(left), 1(right), y: 0(up), 1(down)
        self.direction = [0, 0]
        self.size = size
        self.on_top = on_top
        self.localPos = None
        self.ready = True
        self.w = None
        self.h = None

        self.setupUi()
        self.show()

    def move(self, xy):
        self.setGeometry(xy[0], xy[1], self.w, self.h)

    def setupUi(self):
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                                      if self.on_top
                                      else QtCore.Qt.FramelessWindowHint)
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        label = QtWidgets.QLabel(centralWidget)
        movie = QMovie(self.img_path)
        label.setMovie(movie)
        movie.start()
        movie.stop()

        self.w = int(movie.frameRect().size().width() * self.size)
        self.h = int(movie.frameRect().size().height() * self.size)
        movie.setScaledSize(QtCore.QSize(self.w, self.h))
        movie.start()

        self.setGeometry(self.xy[0], self.xy[1], self.w, self.h)