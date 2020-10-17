from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication, QCheckBox
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread
from PyQt5.Qt import QColor, QPalette
import os
from pyqtgraph import PlotItem, BarGraphItem
from pyqtgraph import GraphicsLayoutWidget
from PyQt5 import uic
import seabreeze.spectrometers as sb
from gui.modules import spectrometers as mock
from tools.threadWorker import Worker
import numpy as np

import logging


log = logging.getLogger(__name__)

filterViewUiPath = os.path.dirname(os.path.realpath(__file__)) + "\\filterViewUi.ui"
Ui_filterView, QtBaseClass = uic.loadUiType(filterViewUiPath)


class FilterView(QWidget, Ui_filterView):
    s_data_changed = pyqtSignal(dict)

    def __init__(self, model=None, controller=None):
        super(FilterView, self).__init__()
        self.model = model
        self.setupUi(self)

        self.acqThread = QThread()
        self.plotItem = None
        self.dataPlotItem = None
        self.acqWorker = None
        self.spec = None
        self.waves = None
        self.exposure_time = 100
        self.integration_count = 1
        self.backgroundData = None
        self.acquisitionData = None
        self.analysedData = None
        self.isNormalized = None

        self.isAcqAlive = 0
        self.deviceConnected = 0

        self.connect_buttons()
        self.connect_signals()
        self.connect_checkbox()
        self.make_threads()
        self.create_plots()
        self.initialize_device()
        self.manage_indicators()

    def initialize_device(self):
        log.debug("Initializing devices...")
        try:
            devices = sb.list_devices()
            self.spec = sb.Spectrometer(devices[0])
            log.info("Devices:{}".format(devices))
            self.deviceConnected = True
        except IndexError as e:
            log.warning("No SpectrumDevice was found. Try connecting manually.")
            self.deviceConnected = False
            self.spec = mock.MockSpectrometer()
            log.info("No device found; Mocking Spectrometer Enabled.")

        self.spec.integration_time_micros(int(float(self.le_exposure.text()) * 1000))

    def connect_buttons(self):
        self.pb_liveView.clicked.connect(self.toggle_live_view)

        self.pb_rmBackground.clicked.connect(lambda: setattr(self, "backgroundData", True))
        self.pb_rmBackground.clicked.connect(self.manage_indicators)

        self.pb_analyse.clicked.connect(lambda: setattr(self, "analysedData", True))
        self.pb_analyse.clicked.connect(lambda: setattr(self, "acquisitionData", True))
        self.pb_analyse.clicked.connect(self.manage_indicators)

        self.pb_normalize.clicked.connect(lambda: setattr(self, "isNormalized", True))
        self.pb_normalize.clicked.connect(lambda: self.manage_indicators())

        self.le_exposure.textChanged.connect(self.set_exposure_time)
        self.le_viewTime.textChanged.connect(self.set_integration_time)

        log.debug("Connecting GUI buttons...")

    def connect_checkbox(self):
        self.ind_rmBackground.clicked.connect(lambda:print("showBackground if available"))
        self.ind_normalize.clicked.connect(lambda: print("show normalisation if available"))
        self.ind_analyse.clicked.connect(lambda: print("show acquisition if available"))

    def connect_signals(self):
        log.debug("Connecting GUI signals...")
        self.s_data_changed.connect(self.update_graph)

    def make_threads(self, *args):
        self.acqWorker = Worker(self.read_data_live, *args)
        self.acqWorker.moveToThread(self.acqThread)
        self.acqThread.started.connect(self.acqWorker.run)

    def create_plots(self):
        log.debug("Creating GUI plots...")
        self.pyqtgraphWidget.clear()
        self.plotItem = self.pyqtgraphWidget.addPlot()
        self.dataPlotItem = self.plotItem.plot()

    def manage_indicators(self):
        if self.isAcqAlive:
            self.ind_rmBackground.setEnabled(True)
            self.ind_normalize.setEnabled(True)
            self.ind_analyse.setEnabled(True)
            self.pb_rmBackground.setEnabled(True)
            self.pb_normalize.setEnabled(True)
            self.pb_analyse.setEnabled(True)
            if self.backgroundData is None:
                self.ind_rmBackground.setStyleSheet("QCheckBox::indicator{background-color: #db1a1a;}")
                try:
                    self.ind_rmBackground.clicked.disconnect()
                except Exception:
                    pass
            else:
                self.ind_rmBackground.setStyleSheet("QCheckBox::indicator{background-color: #55b350;}")

            if self.isNormalized is None:
                self.ind_normalize.setStyleSheet("QCheckBox::indicator{background-color: #db1a1a;}")
                try:
                    self.ind_normalize.clicked.disconnect()
                except Exception:
                    pass
            else:
                self.ind_normalize.setStyleSheet("QCheckBox::indicator{background-color: #55b350;}")

            if self.acquisitionData is None:
                self.ind_analyse.setStyleSheet("QCheckBox::indicator{background-color: #db1a1a;}")
                try:
                    self.ind_analyse.clicked.disconnect()
                except Exception:
                    pass
            else:
                self.ind_analyse.setStyleSheet("QCheckBox::indicator{background-color: #55b350;}")
        else:
            self.pb_rmBackground.setEnabled(False)
            self.pb_normalize.setEnabled(False)
            self.pb_analyse.setEnabled(False)
            self.ind_rmBackground.setEnabled(False)
            self.ind_normalize.setEnabled(False)
            self.ind_analyse.setEnabled(False)
            self.ind_rmBackground.setStyleSheet("QCheckBox::indicator{background-color: #9e9e9e;}")
            self.ind_normalize.setStyleSheet("QCheckBox::indicator{background-color: #9e9e9e;}")
            self.ind_analyse.setStyleSheet("QCheckBox::indicator{background-color: #9e9e9e;}")

    def load_parameters(self):
        self.showRmBackgroundWarning = 1

    # Non-Initializing Functions

    @pyqtSlot(dict)
    def update_graph(self, plotData):
        y = plotData["y"]
        self.dataPlotItem.setData(self.waves, y)

    def read_data_live(self, *args, **kwargs):
        self.waves = self.spec.wavelengths()[2:]

        while self.isAcqAlive:
            intens = []
            for _ in range(self.integration_count):
                intens.append(self.spec.intensities()[2:])
            intens = np.mean(intens, axis=0)
            self.s_data_changed.emit({"y": intens})

    def make_threads(self, *args):
        self.acqWorker = Worker(self.read_data_live, *args)
        self.acqWorker.moveToThread(self.acqThread)
        self.acqThread.started.connect(self.acqWorker.run)

    def toggle_live_view(self):
        if not self.isAcqAlive:
            try:
                self.acqThread.start()
                self.isAcqAlive = True
                self.pb_liveView.start_flash()
            except Exception as e:
                log.error(e)
                self.spec = mock.MockSpectrometer()

        else:
            self.acqThread.terminate()
            self.pb_liveView.stop_flash()
            self.isAcqAlive = False
        self.manage_indicators()

    def set_exposure_time(self, time_in_ms):
        try:
            self.exposure_time = int(time_in_ms)
            self.spec.integration_time_micros(self.exposure_time * 1000)
            self.le_exposure.setStyleSheet('color: black')
        except ValueError as e:
            self.le_exposure.setStyleSheet('color: red')
            log.error(e)

        self.set_integration_time()

    def set_integration_time(self, time_in_ms=None):
        try:
            if not time_in_ms:
                time_in_ms = self.le_viewTime.text()

            integration_time = int(time_in_ms)
            if integration_time >= self.exposure_time:
                self.integration_count = integration_time // self.exposure_time
                self.le_viewTime.setStyleSheet('color: black')
            else:
                self.integration_count = 1
                self.le_viewTime.setStyleSheet('color: red')
                # self.le_viewTime.setText(str(self.exposure_time))
        except ValueError as e:
            log.error(e)
            self.le_viewTime.setStyleSheet('color: red')

    def visualize_any_acquisition(self):
        pass
    def remove_background(self):
        if self.showRmBackgroundWarning:
            warningDialog = QMessageBox()
            warningDialog.setIcon(QMessageBox.Information)
            #warningDialog.setText("Information:")
            warningDialog.setText("Your light source should be 'OFF' before removing the background signal.")
            warningDialog.setWindowTitle("Remove Background")
            warningDialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            doNotShow = QCheckBox("Do not show again.")
            warningDialog.setCheckBox(doNotShow)
            warningDialog.buttonClicked.connect(lambda:print("lol"))
            doNotShow.clicked.connect(lambda: setattr(self, 'showRmBackgroundWarning', 0))
            warningDialog.exec_()

# TODO:
# remove background, normalize (take ref, create norm, norm stream)
# add wavelength line cursor with value display
