from PyQt5 import QtCore, QtGui, QtWidgets
import ast
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PyQt5.QtGui import QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout
import sys
import os
from .reducepilatus_ReduSAXS import SAXSDataSet,EdfFiles,place_legend_outside
import subprocess
import csv
import re
import functools
import types
import traceback
import numpy as np
import fabio
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import shutil
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from pathlib import Path
from pathlib import Path

datapath = str(Path.home())
#datapath="T:/LPCNO/NCO/Manips/DATA_SAXS/SAXS_LGC/"


py311_path = os.path.dirname(sys.executable) + os.sep


# ---------------------------------------------------------------------------
# File selection by logbook sample name, and I_0 / mask auto-fill from the
# files next to the data (same behaviour as SAXSProcessor/saxs_explorer.py).
# ---------------------------------------------------------------------------

# Where the file dialogs open, and the ceiling for the mask search below.
# The package can be installed anywhere, so this is a data location rather
# than the code location: $SAXS_ROOT_DIR if set, else the user's home.
SAXS_ROOT_DIR = os.environ.get("SAXS_ROOT_DIR") or str(Path.home())


def _is_within_saxs_dir(path):
    if not path or not os.path.isdir(path):
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), SAXS_ROOT_DIR]) == SAXS_ROOT_DIR
    except ValueError:
        return False


def default_browse_dir(start_dir):
    """start_dir if it lies inside the SAXS archive, else the SAXS root."""
    return start_dir if _is_within_saxs_dir(start_dir) else SAXS_ROOT_DIR


_LOGBOOK_NUMBER_RE = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
_LOGBOOK_FILENUMBER_RE = re.compile(r'(\d+(?:-\d+)?)\.edf$', re.IGNORECASE)


def find_logbook_i0(data_file):
    """
    The '#Incident beam intensity (I_0), <value>' line of the logbook.csv
    next to data_file, as a float, or None. Takes the first number in the
    cell, since real entries look like "1960000 (tbv)".
    """
    logbook_path = os.path.join(os.path.dirname(data_file), "logbook.csv")
    if not os.path.isfile(logbook_path):
        return None
    try:
        with open(logbook_path, "r", errors="replace") as f:
            for line in f:
                if line.lstrip("#").strip().lower().startswith("incident beam intensity"):
                    match = _LOGBOOK_NUMBER_RE.search(line.split(",", 1)[1])
                    return float(match.group()) if match else None
    except (OSError, IndexError):
        return None
    return None


def parse_logbook_entries(logbook_path):
    """{"File Number": "Comment"} from a logbook.csv, e.g. {"00002-00003": "TDFeNP94-NC-H0"}."""
    entries = {}
    try:
        with open(logbook_path, "r", errors="replace") as f:
            data_lines = [line for line in f if not line.lstrip().startswith("#")]
    except OSError:
        return entries

    reader = csv.DictReader(data_lines)
    if not reader.fieldnames:
        return entries
    fields = {name.strip().lower(): name for name in reader.fieldnames}
    file_number_key = fields.get("file number")
    comment_key = fields.get("comment")
    if file_number_key is None or comment_key is None:
        return entries

    for row in reader:
        file_number = (row.get(file_number_key) or "").strip()
        if file_number:
            entries[file_number] = (row.get(comment_key) or "").strip()
    return entries


def find_logbook_samplenames(data_files):
    """Logbook sample name (or None) for each file, parsing each folder's logbook once."""
    cache = {}
    names = []
    for data_file in data_files:
        folder = os.path.dirname(data_file)
        if folder not in cache:
            logbook_path = os.path.join(folder, "logbook.csv")
            cache[folder] = parse_logbook_entries(logbook_path) if os.path.isfile(logbook_path) else {}
        match = _LOGBOOK_FILENUMBER_RE.search(os.path.basename(data_file))
        names.append((cache[folder].get(match.group(1)) or None) if match else None)
    return names


def find_directory_mask(data_file):
    """
    Nearest .edf with "mask" in its name, searching data_file's folder first
    and then walking up to the SAXS root: one mask usually covers a whole
    beamtime and sits at the run root, above per-sample subfolders. The
    shortest name wins, since variants carry extra suffixes (amask_corr.edf).
    """
    folder = os.path.dirname(os.path.abspath(data_file))
    while True:
        try:
            candidates = sorted(
                (name for name in os.listdir(folder)
                 if name.lower().endswith(".edf") and "mask" in name.lower()),
                key=lambda name: (len(name), name),
            )
        except OSError:
            candidates = []
        if candidates:
            return os.path.join(folder, candidates[0])
        parent = os.path.dirname(folder)
        if parent == folder or not _is_within_saxs_dir(parent):
            return None
        folder = parent


class FileByNameDialog(QtWidgets.QDialog):
    """
    Lists a folder's files alongside their logbook sample name, with a
    filter box and multi-select. The single "Add selected" button (or a
    double-click) returns the highlighted files in self.selected_files and
    closes the dialog; closing it any other way returns nothing.
    """

    def __init__(self, folder, parent=None, single_selection=False, extensions=(".edf",)):
        super().__init__(parent)
        self.folder = folder
        self.selected_files = []
        self.single_selection = single_selection
        self.extensions = tuple(e.lower() for e in extensions)
        verb = "Select a file" if single_selection else "Add files"
        self.setWindowTitle(f"{verb} by name — {folder}")

        layout = QtWidgets.QVBoxLayout(self)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Filter (sample name or file contains)"))
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit, 1)
        self.line_eraser_check = QtWidgets.QCheckBox("Line eraser only")
        self.line_eraser_check.setToolTip(
            "Show only '_vd_' files — the combined, module-gap-corrected "
            "acquisitions — hiding the raw half-frames next to them.")
        # Shown and defaulted from what the folder holds (see _populate):
        # left on in a folder without _vd_ files it would hide every row.
        self.line_eraser_check.setChecked(False)
        self.line_eraser_check.setVisible(False)
        self.line_eraser_check.toggled.connect(lambda _checked: self._apply_filter(self.filter_edit.text()))
        filter_row.addWidget(self.line_eraser_check)
        layout.addLayout(filter_row)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Sample name", "File"])
        self.table.setColumnWidth(0, 260)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection if single_selection
            else QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.itemDoubleClicked.connect(lambda _item: self._on_add_clicked())
        layout.addWidget(self.table, 1)

        button_row = QtWidgets.QHBoxLayout()
        if not single_selection:
            select_all_button = QtWidgets.QPushButton("Select all visible")
            select_all_button.clicked.connect(self._select_all_visible)
            button_row.addWidget(select_all_button)
            select_none_button = QtWidgets.QPushButton("Select none")
            select_none_button.clicked.connect(self.table.clearSelection)
            button_row.addWidget(select_none_button)
        button_row.addStretch(1)
        self.add_button = QtWidgets.QPushButton("Select" if single_selection else "Add selected")
        self.add_button.setDefault(True)
        self.add_button.clicked.connect(self._on_add_clicked)
        self.table.itemSelectionChanged.connect(self._update_add_button)
        button_row.addWidget(self.add_button)
        layout.addLayout(button_row)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(min(int(avail.width() * 0.5), 700), min(int(avail.height() * 0.6), 600))
        else:
            self.resize(650, 500)

        self._populate()
        self._apply_filter(self.filter_edit.text())
        self._update_add_button()

    def _populate(self):
        try:
            names = sorted(n for n in os.listdir(self.folder) if n.lower().endswith(self.extensions))
        except OSError:
            names = []
        # Ordered by filename, not sample name: scan numbers follow the
        # experiment sequence, while names like H100 < H15 sort wrongly.
        sample_names = find_logbook_samplenames([os.path.join(self.folder, n) for n in names])

        self.table.setRowCount(0)
        for sample_name, name in zip(sample_names, names):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(sample_name or "(unknown — no logbook match)"))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(name))

        has_line_eraser = any("vd" in name.split("_") for name in names)
        self.line_eraser_check.setVisible(has_line_eraser)
        self.line_eraser_check.setChecked(has_line_eraser)

    def _apply_filter(self, text):
        text = text.strip().lower()
        line_eraser_only = self.line_eraser_check.isChecked()
        for r in range(self.table.rowCount()):
            filename = self.table.item(r, 1).text()
            visible = text in self.table.item(r, 0).text().lower() or text in filename.lower()
            if line_eraser_only and "vd" not in filename.split("_"):
                visible = False
            self.table.setRowHidden(r, not visible)

    def _select_all_visible(self):
        # selectRow() in a loop replaces the selection each time; one
        # QItemSelection applied at once is what actually selects them all.
        selection = QtCore.QItemSelection()
        for r in range(self.table.rowCount()):
            if not self.table.isRowHidden(r):
                index = self.table.model().index(r, 0)
                selection.select(index, index)
        self.table.selectionModel().select(
            selection, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    def _selected_rows(self):
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def _on_add_clicked(self):
        rows = self._selected_rows()
        if not rows:
            return
        self.selected_files = [os.path.join(self.folder, self.table.item(r, 1).text()) for r in rows]
        self.accept()

    def _update_add_button(self):
        n = len(self._selected_rows())
        self.add_button.setEnabled(n > 0)
        if not self.single_selection:
            self.add_button.setText(f"Add {n} selected file(s)" if n else "Add selected")


def pick_files_by_name(parent, start_dir, single_selection=False):
    """Ask for a folder, then pick files in it by sample name. Returns (folder, files)."""
    folder = QtWidgets.QFileDialog.getExistingDirectory(parent, "Select folder", default_browse_dir(start_dir))
    if not folder:
        return None, []
    dialog = FileByNameDialog(folder, parent=parent, single_selection=single_selection)
    dialog.exec_()
    return folder, dialog.selected_files

# Cycled over the sector directions (see _sector_curves).
SECTOR_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))]


def sample_colors(n):
    """One colour per sample. The default cycle only holds 10 before repeating,
    so beyond that the colours are spread along turbo instead."""
    if n <= 10:
        return [f"C{i}" for i in range(n)]
    return list(plt.cm.turbo(np.linspace(0.05, 0.95, n)))


def format_q_coord(x, y):
    """Cursor readout for an I(q) plot: q, the matching real-space distance, and I."""
    if x is None or y is None:
        return ""
    text = f"q={x:.4f} A-1"
    if x > 0:
        text += f"   d={2 * np.pi / x / 10:.3f} nm"
    return text + f"   I={y:.4g}"


def format_chi_coord(x, y):
    """Cursor readout for an I(chi) plot."""
    if x is None or y is None:
        return ""
    return f"chi={x:.2f}deg   I={y:.4g}"


def report_errors(method):
    """Show an exception raised by a button handler in the console and a message box instead of killing the app."""
    @functools.wraps(method)
    def wrapper(self, *_signal_args):
        try:
            return method(self)
        except Exception as exc:
            traceback.print_exc()
            self._log(f"ERROR: {exc}")
            QtWidgets.QMessageBox.warning(self, "SAXS toolbox", str(exc))
    return wrapper

class MplWidget(QWidget):
    def __init__(self, parent=None):
        
        super(MplWidget, self).__init__(parent)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)  
        self.layout = QVBoxLayout()
        self.canvas.axes = self.fig.add_subplot(111) 
        self.setLayout(self.layout)

        # In the layout rather than floating over the canvas. coordinates=False
        # drops its readout label: the cursor position goes to the window's
        # status bar instead, which stays visible however narrow the window is.
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=False)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.setMaximumHeight(420)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self.setSizePolicy(sizePolicy)

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_new.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.



class Ui_SAXS_TOOLBOX(object):
    def setupUi(self, SAXS_TOOLBOX):
        SAXS_TOOLBOX.setObjectName("SAXS_TOOLBOX")
        SAXS_TOOLBOX.resize(1106, 911)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SAXS_TOOLBOX.sizePolicy().hasHeightForWidth())
        SAXS_TOOLBOX.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        SAXS_TOOLBOX.setFont(font)
        self.centralwidget = QtWidgets.QWidget(SAXS_TOOLBOX)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.tab_3)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.data_file_textEdit = QtWidgets.QTextEdit(self.tab_3)
        self.data_file_textEdit.setObjectName("data_file_textEdit")
        self.horizontalLayout_2.addWidget(self.data_file_textEdit)
        self.data_file_pushButton = QtWidgets.QPushButton(self.tab_3)
        self.data_file_pushButton.setObjectName("data_file_pushButton")
        self.horizontalLayout_2.addWidget(self.data_file_pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.nb_samples_averaged_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.nb_samples_averaged_lineEdit.setObjectName("nb_samples_averaged_lineEdit")
        self.horizontalLayout_4.addWidget(self.nb_samples_averaged_lineEdit)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem7)
        self.save_avge_edf_checkBox = QtWidgets.QCheckBox(self.tab_3)
        self.save_avge_edf_checkBox.setObjectName("save_avge_edf_checkBox")
        self.horizontalLayout_4.addWidget(self.save_avge_edf_checkBox)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        spacerItemData = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItemData)
        self.label_33 = QtWidgets.QLabel(self.tab_3)
        self.label_33.setObjectName("label_33")
        self.verticalLayout.addWidget(self.label_33)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.mask_file_lineEdit_2 = QtWidgets.QLineEdit(self.tab_3)
        self.mask_file_lineEdit_2.setObjectName("mask_file_lineEdit_2")
        self.horizontalLayout_22.addWidget(self.mask_file_lineEdit_2)
        self.mask_file_pushButton_2 = QtWidgets.QPushButton(self.tab_3)
        self.mask_file_pushButton_2.setObjectName("mask_file_pushButton_2")
        self.horizontalLayout_22.addWidget(self.mask_file_pushButton_2)
        self.pyFAImask_checkBox = QtWidgets.QCheckBox(self.tab_3)
        self.pyFAImask_checkBox.setObjectName("pyFAImask_checkBox")
        self.horizontalLayout_22.addWidget(self.pyFAImask_checkBox)
        self.verticalLayout.addLayout(self.horizontalLayout_22)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout.addLayout(self.horizontalLayout_3)
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.I0_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.I0_lineEdit.setObjectName("I0_lineEdit")
        self.horizontalLayout.addWidget(self.I0_lineEdit)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.horizontalLayout.addLayout(self.horizontalLayout_6)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.force_distance_checkBox = QtWidgets.QCheckBox(self.tab_3)
        self.force_distance_checkBox.setObjectName("force_distance_checkBox")
        self.horizontalLayout.addWidget(self.force_distance_checkBox)
        self.force_distance_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.force_distance_lineEdit.setObjectName("force_distance_lineEdit")
        self.force_distance_lineEdit.setEnabled(False)
        self.force_distance_lineEdit.setMaximumWidth(80)
        self.horizontalLayout.addWidget(self.force_distance_lineEdit)
        self.label_force_distance_unit = QtWidgets.QLabel(self.tab_3)
        self.label_force_distance_unit.setObjectName("label_force_distance_unit")
        self.horizontalLayout.addWidget(self.label_force_distance_unit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.label_7 = QtWidgets.QLabel(self.tab_3)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.ref_file_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.ref_file_lineEdit.setObjectName("ref_file_lineEdit")
        self.horizontalLayout_5.addWidget(self.ref_file_lineEdit)
        self.ref_file_pushButton = QtWidgets.QPushButton(self.tab_3)
        self.ref_file_pushButton.setObjectName("ref_file_pushButton")
        self.horizontalLayout_5.addWidget(self.ref_file_pushButton)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.ref_Multiplier_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.ref_Multiplier_lineEdit.setObjectName("ref_Multiplier_lineEdit")
        self.horizontalLayout_5.addWidget(self.ref_Multiplier_lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.ref_corr_checkBox = QtWidgets.QCheckBox(self.tab_3)
        self.ref_corr_checkBox.setObjectName("ref_corr_checkBox")
        self.horizontalLayout_18.addWidget(self.ref_corr_checkBox)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem5)
        self.label_28 = QtWidgets.QLabel(self.tab_3)
        self.label_28.setObjectName("label_28")
        self.horizontalLayout_18.addWidget(self.label_28)
        self.qmin_lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.qmin_lineEdit.setObjectName("qmin_lineEdit")
        self.horizontalLayout_18.addWidget(self.qmin_lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_18)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem6)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout.addLayout(self.verticalLayout_11)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(self.tab_1)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.groupBox = QtWidgets.QGroupBox(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.water_checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.water_checkBox.setObjectName("water_checkBox")
        self.verticalLayout_18.addWidget(self.water_checkBox)
        self.hexane_checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.hexane_checkBox.setObjectName("hexane_checkBox")
        self.verticalLayout_18.addWidget(self.hexane_checkBox)
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_18.addWidget(self.label_12)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_18.addWidget(self.lineEdit)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.label_21 = QtWidgets.QLabel(self.groupBox_2)
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_25.addWidget(self.label_21)
        self.water_ref_calib_file_lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.water_ref_calib_file_lineEdit.setObjectName("water_ref_calib_file_lineEdit")
        self.horizontalLayout_25.addWidget(self.water_ref_calib_file_lineEdit)
        self.water_ref_calib_file_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.water_ref_calib_file_pushButton.setObjectName("water_ref_calib_file_pushButton")
        self.horizontalLayout_25.addWidget(self.water_ref_calib_file_pushButton)
        self.verticalLayout_4.addLayout(self.horizontalLayout_25)
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.label_23 = QtWidgets.QLabel(self.groupBox_2)
        self.label_23.setObjectName("label_23")
        self.horizontalLayout_27.addWidget(self.label_23)
        self.water_calib_file_lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.water_calib_file_lineEdit.setObjectName("water_calib_file_lineEdit")
        self.horizontalLayout_27.addWidget(self.water_calib_file_lineEdit)
        self.water_calib_file_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.water_calib_file_pushButton.setObjectName("water_calib_file_pushButton")
        self.horizontalLayout_27.addWidget(self.water_calib_file_pushButton)
        self.verticalLayout_4.addLayout(self.horizontalLayout_27)
        self.verticalLayout_17.addLayout(self.verticalLayout_4)
        self.verticalLayout_18.addWidget(self.groupBox_2)
        self.verticalLayout_19.addWidget(self.groupBox)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_19.addItem(spacerItem9)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_10 = QtWidgets.QWidget()
        self.tab_10.setObjectName("tab_10")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab_10)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 1081, 601))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.radial_groupBox = QtWidgets.QGroupBox(self.verticalLayoutWidget_2)
        self.radial_groupBox.setObjectName("radial_groupBox")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.radial_groupBox)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.radial_azimuth_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.horizontalLayout_13.addWidget(self.radial_azimuth_label)
        self.radial_azimuth_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_2)
        self.radial_azimuth_doubleSpinBox.setObjectName("radial_azimuth_doubleSpinBox")
        self.radial_azimuth_doubleSpinBox.setRange(-180.0, 180.0)
        self.radial_azimuth_doubleSpinBox.setDecimals(1)
        self.horizontalLayout_13.addWidget(self.radial_azimuth_doubleSpinBox)
        self.radial_width_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.horizontalLayout_13.addWidget(self.radial_width_label)
        self.radial_width_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.verticalLayoutWidget_2)
        self.radial_width_doubleSpinBox.setObjectName("radial_width_doubleSpinBox")
        self.radial_width_doubleSpinBox.setRange(0.1, 360.0)
        self.radial_width_doubleSpinBox.setDecimals(1)
        self.horizontalLayout_13.addWidget(self.radial_width_doubleSpinBox)
        self.radial_qmin_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.horizontalLayout_13.addWidget(self.radial_qmin_label)
        self.radial_qmin_display_lineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.radial_qmin_display_lineEdit.setObjectName("radial_qmin_display_lineEdit")
        self.radial_qmin_display_lineEdit.setMaximumWidth(80)
        self.horizontalLayout_13.addWidget(self.radial_qmin_display_lineEdit)
        self.radial_qmax_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.horizontalLayout_13.addWidget(self.radial_qmax_label)
        self.radial_qmax_display_lineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.radial_qmax_display_lineEdit.setObjectName("radial_qmax_display_lineEdit")
        self.radial_qmax_display_lineEdit.setMaximumWidth(80)
        self.horizontalLayout_13.addWidget(self.radial_qmax_display_lineEdit)
        self.horizontalLayout_13.addStretch(1)
        self.radial_regrouppushButton = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.radial_regrouppushButton.setObjectName("radial_regrouppushButton")
        self.horizontalLayout_13.addWidget(self.radial_regrouppushButton)
        self.radial_save_pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.radial_save_pushButton.setObjectName("radial_save_pushButton")
        self.horizontalLayout_13.addWidget(self.radial_save_pushButton)
        self.verticalLayout_12.addWidget(self.radial_groupBox)
        self.radial_plot_frame = QtWidgets.QGroupBox(self.verticalLayoutWidget_2)
        self.radial_plot_frame.setObjectName("radial_plot_frame")
        self.radial_plot_layout = QtWidgets.QVBoxLayout(self.radial_plot_frame)
        self.radial_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.MplWidget = MplWidget(self.radial_plot_frame)
        self.MplWidget.setObjectName("MplWidget")
        self.radial_plot_layout.addWidget(self.MplWidget)
        self.verticalLayout_12.addWidget(self.radial_plot_frame)
        self.verticalLayout_12.addStretch(1)
        self.tabWidget.addTab(self.tab_10, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1081, 591))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_32 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_32.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_32.setObjectName("verticalLayout_32")
        self.sector_params_widget = QtWidgets.QWidget()
        self.sector_params_widget.setObjectName("sector_params_widget")
        self.horizontalLayout_44 = QtWidgets.QHBoxLayout(self.sector_params_widget)
        self.horizontalLayout_44.setObjectName("horizontalLayout_44")
        self.label_47 = QtWidgets.QLabel(self.sector_params_widget)
        self.label_47.setObjectName("label_47")
        self.horizontalLayout_44.addWidget(self.label_47)
        self.number_azimuthspinBox = QtWidgets.QSpinBox(self.sector_params_widget)
        self.number_azimuthspinBox.setObjectName("number_azimuthspinBox")
        self.horizontalLayout_44.addWidget(self.number_azimuthspinBox)
        self.label_48 = QtWidgets.QLabel(self.sector_params_widget)
        self.label_48.setObjectName("label_48")
        self.horizontalLayout_44.addWidget(self.label_48)
        self.angle_spinBox = QtWidgets.QSpinBox(self.sector_params_widget)
        self.angle_spinBox.setObjectName("angle_spinBox")
        self.angle_spinBox.setRange(0, 360)
        self.horizontalLayout_44.addWidget(self.angle_spinBox)
        self.label_49 = QtWidgets.QLabel(self.sector_params_widget)
        self.label_49.setObjectName("label_49")
        self.horizontalLayout_44.addWidget(self.label_49)
        self.offset_lineEdit = QtWidgets.QLineEdit(self.sector_params_widget)
        self.offset_lineEdit.setObjectName("offset_lineEdit")
        self.offset_lineEdit.setMaximumWidth(80)
        self.horizontalLayout_44.addWidget(self.offset_lineEdit)
        self.label_50 = QtWidgets.QLabel(self.sector_params_widget)
        self.label_50.setObjectName("label_50")
        self.horizontalLayout_44.addWidget(self.label_50)
        self.width_lineEdit = QtWidgets.QLineEdit(self.sector_params_widget)
        self.width_lineEdit.setObjectName("width_lineEdit")
        self.width_lineEdit.setMaximumWidth(80)
        self.horizontalLayout_44.addWidget(self.width_lineEdit)
        self.sector_qmin_label = QtWidgets.QLabel(self.sector_params_widget)
        self.horizontalLayout_44.addWidget(self.sector_qmin_label)
        self.sector_qmin_display_lineEdit = QtWidgets.QLineEdit(self.sector_params_widget)
        self.sector_qmin_display_lineEdit.setObjectName("sector_qmin_display_lineEdit")
        self.sector_qmin_display_lineEdit.setMaximumWidth(80)
        self.horizontalLayout_44.addWidget(self.sector_qmin_display_lineEdit)
        self.sector_qmax_label = QtWidgets.QLabel(self.sector_params_widget)
        self.horizontalLayout_44.addWidget(self.sector_qmax_label)
        self.sector_qmax_display_lineEdit = QtWidgets.QLineEdit(self.sector_params_widget)
        self.sector_qmax_display_lineEdit.setObjectName("sector_qmax_display_lineEdit")
        self.sector_qmax_display_lineEdit.setMaximumWidth(80)
        self.horizontalLayout_44.addWidget(self.sector_qmax_display_lineEdit)
        self.horizontalLayout_44.addStretch(1)
        self.sector_regrouppushButton = QtWidgets.QPushButton(self.sector_params_widget)
        self.sector_regrouppushButton.setObjectName("sector_regrouppushButton")
        self.horizontalLayout_44.addWidget(self.sector_regrouppushButton)
        self.sector_save_pushButton = QtWidgets.QPushButton(self.sector_params_widget)
        self.sector_save_pushButton.setObjectName("sector_save_pushButton")
        self.horizontalLayout_44.addWidget(self.sector_save_pushButton)
        self.azim_params_widget = QtWidgets.QWidget()
        self.azim_params_widget.setObjectName("azim_params_widget")
        self.horizontalLayout_45 = QtWidgets.QHBoxLayout(self.azim_params_widget)
        self.horizontalLayout_45.setObjectName("horizontalLayout_45")
        self.azim_q_label = QtWidgets.QLabel(self.azim_params_widget)
        self.horizontalLayout_45.addWidget(self.azim_q_label)
        self.azim_q_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.azim_params_widget)
        self.azim_q_doubleSpinBox.setObjectName("azim_q_doubleSpinBox")
        self.azim_q_doubleSpinBox.setDecimals(4)
        self.azim_q_doubleSpinBox.setRange(0.0001, 5.0)
        self.azim_q_doubleSpinBox.setSingleStep(0.001)
        self.horizontalLayout_45.addWidget(self.azim_q_doubleSpinBox)
        self.azim_threshold_label = QtWidgets.QLabel(self.azim_params_widget)
        self.horizontalLayout_45.addWidget(self.azim_threshold_label)
        self.azim_threshold_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.azim_params_widget)
        self.azim_threshold_doubleSpinBox.setObjectName("azim_threshold_doubleSpinBox")
        self.azim_threshold_doubleSpinBox.setDecimals(5)
        self.azim_threshold_doubleSpinBox.setRange(0.00001, 1.0)
        self.azim_threshold_doubleSpinBox.setSingleStep(0.0005)
        self.horizontalLayout_45.addWidget(self.azim_threshold_doubleSpinBox)
        self.horizontalLayout_45.addStretch(1)
        self.azim_plot_pushButton = QtWidgets.QPushButton(self.azim_params_widget)
        self.azim_plot_pushButton.setObjectName("azim_plot_pushButton")
        self.horizontalLayout_45.addWidget(self.azim_plot_pushButton)
        self.azim_save_pushButton = QtWidgets.QPushButton(self.azim_params_widget)
        self.azim_save_pushButton.setObjectName("azim_save_pushButton")
        self.horizontalLayout_45.addWidget(self.azim_save_pushButton)
        self.sector_plot_tabWidget = QtWidgets.QTabWidget(self.verticalLayoutWidget)
        self.sector_plot_tabWidget.setObjectName("sector_plot_tabWidget")
        self.sector_plot_page = QtWidgets.QWidget()
        self.sector_plot_page_layout = QtWidgets.QVBoxLayout(self.sector_plot_page)
        self.sector_plot_page_layout.setContentsMargins(0, 0, 0, 0)
        self.sector_plot_page_layout.addWidget(self.sector_params_widget)
        self.MplWidget_4 = MplWidget(self.sector_plot_page)
        self.MplWidget_4.setObjectName("MplWidget_4")
        self.sector_plot_page_layout.addWidget(self.MplWidget_4)
        self.sector_plot_page_layout.addStretch(1)
        self.sector_plot_tabWidget.addTab(self.sector_plot_page, "")
        self.azim_plot_page = QtWidgets.QWidget()
        self.azim_plot_page_layout = QtWidgets.QVBoxLayout(self.azim_plot_page)
        self.azim_plot_page_layout.setContentsMargins(0, 0, 0, 0)
        self.azim_plot_page_layout.addWidget(self.azim_params_widget)
        self.MplWidget_5 = MplWidget(self.azim_plot_page)
        self.MplWidget_5.setObjectName("MplWidget_5")
        self.azim_plot_page_layout.addWidget(self.MplWidget_5)
        self.azim_plot_page_layout.addStretch(1)
        self.sector_plot_tabWidget.addTab(self.azim_plot_page, "")
        self.verticalLayout_32.addWidget(self.sector_plot_tabWidget)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.image_tab_layout = QtWidgets.QHBoxLayout(self.tab_2)
        # Left: which images
        self.image_files_groupBox = QtWidgets.QGroupBox(self.tab_2)
        self.image_files_groupBox.setObjectName("image_files_groupBox")
        self.image_files_groupBox.setMinimumWidth(300)
        self.image_files_groupBox.setMaximumWidth(380)
        self.image_files_layout = QtWidgets.QVBoxLayout(self.image_files_groupBox)
        self.image_listWidget = QtWidgets.QListWidget(self.image_files_groupBox)
        self.image_listWidget.setObjectName("image_listWidget")
        self.image_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.image_files_layout.addWidget(self.image_listWidget)
        self.image_save_row = QtWidgets.QHBoxLayout()
        self.saveSelectedImages_pushButton = QtWidgets.QPushButton(self.image_files_groupBox)
        self.saveSelectedImages_pushButton.setObjectName("saveSelectedImages_pushButton")
        self.image_save_row.addWidget(self.saveSelectedImages_pushButton)
        self.browseimage_pushButton_2 = QtWidgets.QPushButton(self.image_files_groupBox)
        self.browseimage_pushButton_2.setObjectName("browseimage_pushButton_2")
        self.image_save_row.addWidget(self.browseimage_pushButton_2)
        self.image_files_layout.addLayout(self.image_save_row)
        self.image_tab_layout.addWidget(self.image_files_groupBox)
        # Right: parameters, then preview
        self.image_right_layout = QtWidgets.QVBoxLayout()
        self.image_params_groupBox = QtWidgets.QGroupBox(self.tab_2)
        self.image_params_groupBox.setObjectName("image_params_groupBox")
        self.image_params_grid = QtWidgets.QGridLayout(self.image_params_groupBox)
        self.label_10 = QtWidgets.QLabel(self.image_params_groupBox)
        self.image_params_grid.addWidget(self.label_10, 0, 0)
        self.qminplot_lineEdit = QtWidgets.QLineEdit(self.image_params_groupBox)
        self.qminplot_lineEdit.setObjectName("qminplot_lineEdit")
        self.image_params_grid.addWidget(self.qminplot_lineEdit, 0, 1)
        self.label_11 = QtWidgets.QLabel(self.image_params_groupBox)
        self.image_params_grid.addWidget(self.label_11, 0, 2)
        self.qmaxplot_lineEdit = QtWidgets.QLineEdit(self.image_params_groupBox)
        self.qmaxplot_lineEdit.setObjectName("qmaxplot_lineEdit")
        self.image_params_grid.addWidget(self.qmaxplot_lineEdit, 0, 3)
        self.label_13 = QtWidgets.QLabel(self.image_params_groupBox)
        self.image_params_grid.addWidget(self.label_13, 0, 4)
        self.qcircles_lineEdit = QtWidgets.QLineEdit(self.image_params_groupBox)
        self.qcircles_lineEdit.setObjectName("qcircles_lineEdit")
        self.image_params_grid.addWidget(self.qcircles_lineEdit, 0, 5)
        self.label_4 = QtWidgets.QLabel(self.image_params_groupBox)
        self.image_params_grid.addWidget(self.label_4, 1, 0)
        self.vmin_lineEdit = QtWidgets.QLineEdit(self.image_params_groupBox)
        self.vmin_lineEdit.setObjectName("vmin_lineEdit")
        self.image_params_grid.addWidget(self.vmin_lineEdit, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.image_params_groupBox)
        self.image_params_grid.addWidget(self.label_6, 1, 2)
        self.vmax_lineEdit = QtWidgets.QLineEdit(self.image_params_groupBox)
        self.vmax_lineEdit.setObjectName("vmax_lineEdit")
        self.image_params_grid.addWidget(self.vmax_lineEdit, 1, 3)
        self.label_8 = QtWidgets.QLabel(self.image_params_groupBox)
        self.image_params_grid.addWidget(self.label_8, 1, 4)
        self.cmap_comboBox = QtWidgets.QComboBox(self.image_params_groupBox)
        self.cmap_comboBox.setEditable(False)
        self.cmap_comboBox.setObjectName("cmap_comboBox")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.image_params_grid.addWidget(self.cmap_comboBox, 1, 5)
        self.image_options_row = QtWidgets.QHBoxLayout()
        self.drawSectors_checkBox = QtWidgets.QCheckBox(self.image_params_groupBox)
        self.drawSectors_checkBox.setObjectName("drawSectors_checkBox")
        self.image_options_row.addWidget(self.drawSectors_checkBox)
        self.checkBox = QtWidgets.QCheckBox(self.image_params_groupBox)
        self.checkBox.setObjectName("checkBox")
        self.image_options_row.addWidget(self.checkBox)
        self.image_options_row.addStretch(1)
        self.show_checkBox = QtWidgets.QCheckBox(self.image_params_groupBox)
        self.show_checkBox.setObjectName("show_checkBox")
        self.image_options_row.addWidget(self.show_checkBox)
        self.saveNpz_checkBox = QtWidgets.QCheckBox(self.image_params_groupBox)
        self.saveNpz_checkBox.setObjectName("saveNpz_checkBox")
        self.image_options_row.addWidget(self.saveNpz_checkBox)
        self.image_params_grid.addLayout(self.image_options_row, 2, 0, 1, 6)
        self.image_params_grid.setColumnStretch(5, 1)
        self.image_right_layout.addWidget(self.image_params_groupBox)
        self.image_preview_tabWidget = QtWidgets.QTabWidget(self.tab_2)
        self.image_preview_tabWidget.setObjectName("image_preview_tabWidget")
        self.image_preview_page = QtWidgets.QWidget()
        self.image_preview_layout = QtWidgets.QVBoxLayout(self.image_preview_page)
        self.image_preview_layout.setContentsMargins(0, 0, 0, 0)
        self.image_preview_canvas = FigureCanvas(Figure())
        self.image_preview_canvas.setMinimumHeight(380)
        self.image_preview_toolbar = NavigationToolbar(self.image_preview_canvas, self.image_preview_page, coordinates=False)
        self.image_preview_layout.addWidget(self.image_preview_toolbar)
        self.image_preview_layout.addWidget(self.image_preview_canvas)
        self.image_preview_tabWidget.addTab(self.image_preview_page, "")
        self.image_right_layout.addWidget(self.image_preview_tabWidget, 1)
        self.image_tab_layout.addLayout(self.image_right_layout, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout_9.addWidget(self.tabWidget)
        self.console_textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.console_textEdit.setObjectName("console_textEdit")
        self.verticalLayout_9.addWidget(self.console_textEdit)
        SAXS_TOOLBOX.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SAXS_TOOLBOX)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1106, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menupyFAI = QtWidgets.QMenu(self.menubar)
        self.menupyFAI.setObjectName("menupyFAI")
        self.menuGenerate_mask = QtWidgets.QMenu(self.menupyFAI)
        self.menuGenerate_mask.setObjectName("menuGenerate_mask")
        self.menuInspect_data = QtWidgets.QMenu(self.menubar)
        self.menuInspect_data.setObjectName("menuInspect_data")
        self.menuplot_1D_data_file_s_2 = QtWidgets.QMenu(self.menuInspect_data)
        self.menuplot_1D_data_file_s_2.setObjectName("menuplot_1D_data_file_s_2")
        self.menuSimulation = QtWidgets.QMenu(self.menubar)
        self.menuSimulation.setObjectName("menuSimulation")
        self.menuedf_tools = QtWidgets.QMenu(self.menubar)
        self.menuedf_tools.setObjectName("menuedf_tools")
        SAXS_TOOLBOX.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SAXS_TOOLBOX)
        self.statusbar.setObjectName("statusbar")
        SAXS_TOOLBOX.setStatusBar(self.statusbar)
        self.actionSet_Working_Directory = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionSet_Working_Directory.setObjectName("actionSet_Working_Directory")
        self.actionQuit = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionQuit.setObjectName("actionQuit")
        self.actionInitialize_Form = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionInitialize_Form.setObjectName("actionInitialize_Form")
        self.actionView_image = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionView_image.setObjectName("actionView_image")
        self.actionOpen_view_image = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionOpen_view_image.setObjectName("actionOpen_view_image")
        self.actionDetector_calibration = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionDetector_calibration.setObjectName("actionDetector_calibration")
        self.actionFrame_integration = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionFrame_integration.setObjectName("actionFrame_integration")
        self.actionSWING = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionSWING.setObjectName("actionSWING")
        self.actionfrom_ID15 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionfrom_ID15.setObjectName("actionfrom_ID15")
        self.actionfrom_scratch = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionfrom_scratch.setObjectName("actionfrom_scratch")
        self.actionconvert_foxtrot_mask = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionconvert_foxtrot_mask.setObjectName("actionconvert_foxtrot_mask")
        self.actionAverage_frames = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionAverage_frames.setObjectName("actionAverage_frames")
        self.actionloglog = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionloglog.setObjectName("actionloglog")
        self.actionlinear = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionlinear.setObjectName("actionlinear")
        self.actionOpen_view_image_2 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionOpen_view_image_2.setObjectName("actionOpen_view_image_2")
        self.actionloglog_2 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionloglog_2.setObjectName("actionloglog_2")
        self.actionlinear_2 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionlinear_2.setObjectName("actionlinear_2")
        self.actionAverage_frames_2 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionAverage_frames_2.setObjectName("actionAverage_frames_2")
        self.actionSWING_2 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionSWING_2.setObjectName("actionSWING_2")
        self.actionfrom_ID15_2 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionfrom_ID15_2.setObjectName("actionfrom_ID15_2")
        self.actionfrom_ID02_SAXS = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionfrom_ID02_SAXS.setObjectName("actionfrom_ID02_SAXS")
        self.actionfrom_ID02_WAXS = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionfrom_ID02_WAXS.setObjectName("actionfrom_ID02_WAXS")
        self.actionBatch_radial_profile_plot = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionBatch_radial_profile_plot.setObjectName("actionBatch_radial_profile_plot")
        self.actionCave_frames = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionCave_frames.setObjectName("actionCave_frames")
        self.actionCompute_I_q_from_pdb_file = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionCompute_I_q_from_pdb_file.setObjectName("actionCompute_I_q_from_pdb_file")
        self.actionbatch_horizontal_regrouping = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionbatch_horizontal_regrouping.setObjectName("actionbatch_horizontal_regrouping")
        self.actionbatch_vertical_regrouping = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionbatch_vertical_regrouping.setObjectName("actionbatch_vertical_regrouping")
        self.actionIq_from_pdb = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionIq_from_pdb.setObjectName("actionIq_from_pdb")
        self.actionGenerate_mask = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionGenerate_mask.setObjectName("actionGenerate_mask")
        self.actionconvert_edf_to_png = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionconvert_edf_to_png.setObjectName("actionconvert_edf_to_png")
        self.actionconvert_edf_to_png_2 = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionconvert_edf_to_png_2.setObjectName("actionconvert_edf_to_png_2")
        self.actionaverage_edf_files = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionaverage_edf_files.setObjectName("actionaverage_edf_files")
        self.actionaverage_and_convert_to_png = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actionaverage_and_convert_to_png.setObjectName("actionaverage_and_convert_to_png")
        self.actioncompute_offset = QtWidgets.QAction(SAXS_TOOLBOX)
        self.actioncompute_offset.setObjectName("actioncompute_offset")
        self.menuFile.addAction(self.actionSet_Working_Directory)
        self.menuFile.addAction(self.actionInitialize_Form)
        self.menuGenerate_mask.addAction(self.actionGenerate_mask)
        self.menupyFAI.addAction(self.menuGenerate_mask.menuAction())
        self.menuplot_1D_data_file_s_2.addAction(self.actionloglog_2)
        self.menuplot_1D_data_file_s_2.addAction(self.actionlinear_2)
        self.menuInspect_data.addAction(self.actionOpen_view_image_2)
        self.menuInspect_data.addAction(self.menuplot_1D_data_file_s_2.menuAction())
        self.menuSimulation.addAction(self.actionIq_from_pdb)
        self.menuedf_tools.addAction(self.actionconvert_edf_to_png_2)
        self.menuedf_tools.addAction(self.actionaverage_edf_files)
        self.menuedf_tools.addAction(self.actioncompute_offset)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menupyFAI.menuAction())
        self.menubar.addAction(self.menuInspect_data.menuAction())
        self.menubar.addAction(self.menuSimulation.menuAction())
        self.menubar.addAction(self.menuedf_tools.menuAction())

        self.retranslateUi(SAXS_TOOLBOX)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(SAXS_TOOLBOX)

    def retranslateUi(self, SAXS_TOOLBOX):
        _translate = QtCore.QCoreApplication.translate
        SAXS_TOOLBOX.setWindowTitle(_translate("SAXS_TOOLBOX", "SAXS_TOOLBOX"))
        self.label_33.setText(_translate("SAXS_TOOLBOX", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Load Mask file (.edf)</span></p></body></html>"))
        self.mask_file_pushButton_2.setText(_translate("SAXS_TOOLBOX", "..."))
        self.pyFAImask_checkBox.setText(_translate("SAXS_TOOLBOX", "check if mask obtained from pyFAI (must be inverted)"))
        self.label_2.setText(_translate("SAXS_TOOLBOX", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">I_0</span></p></body></html>"))
        self.force_distance_checkBox.setText(_translate("SAXS_TOOLBOX", "Force sample distance"))
        self.label_force_distance_unit.setText(_translate("SAXS_TOOLBOX", "m"))
        self.label_7.setText(_translate("SAXS_TOOLBOX", "<html><head/><body><p><span style=\" font-weight:600;\">Reference </span>scattering data</p></body></html>"))
        self.ref_file_pushButton.setText(_translate("SAXS_TOOLBOX", "..."))
        self.label_3.setText(_translate("SAXS_TOOLBOX", "Ref. Multiplier"))
        self.ref_corr_checkBox.setText(_translate("SAXS_TOOLBOX", "Refine reference subtraction"))
        self.label_28.setText(_translate("SAXS_TOOLBOX", "q_min for adjusting reference subtraction (angströms-1)"))
        self.label.setText(_translate("SAXS_TOOLBOX", "<html><head/><body><p><span style=\" font-weight:600;\">Specimen </span>scattering data (multiple selection possible)</p></body></html>"))
        self.data_file_pushButton.setText(_translate("SAXS_TOOLBOX", "..."))
        self.label_5.setText(_translate("SAXS_TOOLBOX", "Group & Average by bunch of:"))
        self.save_avge_edf_checkBox.setText(_translate("SAXS_TOOLBOX", "save averaged data as edf Files"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("SAXS_TOOLBOX", "Process data"))
        self.groupBox.setTitle(_translate("SAXS_TOOLBOX", "Nature of the calibrant"))
        self.water_checkBox.setText(_translate("SAXS_TOOLBOX", "Water"))
        self.hexane_checkBox.setText(_translate("SAXS_TOOLBOX", "Hexane"))
        self.label_12.setText(_translate("SAXS_TOOLBOX", "OR specify calibrant Theoretical SLD (cm-1)"))
        self.groupBox_2.setTitle(_translate("SAXS_TOOLBOX", "Load calibration related data files"))
        self.label_21.setText(_translate("SAXS_TOOLBOX", "Reference data (empty cell)"))
        self.water_ref_calib_file_pushButton.setText(_translate("SAXS_TOOLBOX", "..."))
        self.label_23.setText(_translate("SAXS_TOOLBOX", "Calibrant data (cell+sample)"))
        self.water_calib_file_pushButton.setText(_translate("SAXS_TOOLBOX", "..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("SAXS_TOOLBOX", "Intensity calibration"))
        self.radial_groupBox.setTitle(_translate("SAXS_TOOLBOX", "Radial regrouping I(q)"))
        self.radial_azimuth_label.setText(_translate("SAXS_TOOLBOX", "Azimuth (°)"))
        self.radial_width_label.setText(_translate("SAXS_TOOLBOX", "Full width (°)"))
        self.radial_width_doubleSpinBox.setToolTip(_translate("SAXS_TOOLBOX", "Full angular width of the sector centred on the azimuth. 360° = whole ring."))
        self.radial_qmin_label.setText(_translate("SAXS_TOOLBOX", "q min (Å⁻¹)"))
        self.radial_qmax_label.setText(_translate("SAXS_TOOLBOX", "q max (Å⁻¹)"))
        self.radial_qmin_display_lineEdit.setToolTip(_translate("SAXS_TOOLBOX", "Plot display range only \u2014 it does not change the saved .dat files. Leave empty for the full range."))
        self.radial_qmax_display_lineEdit.setToolTip(_translate("SAXS_TOOLBOX", "Plot display range only \u2014 it does not change the saved .dat files. Leave empty for the full range."))
        self.radial_regrouppushButton.setText(_translate("SAXS_TOOLBOX", "Plot"))
        self.radial_save_pushButton.setText(_translate("SAXS_TOOLBOX", "Save"))
        self.radial_save_pushButton.setToolTip(_translate("SAXS_TOOLBOX", "Write I(q) .dat files to radial_profiles/ next to the data"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_10), _translate("SAXS_TOOLBOX", "Radial Regrouping"))
        self.label_47.setText(_translate("SAXS_TOOLBOX", "Directions"))
        self.label_48.setText(_translate("SAXS_TOOLBOX", "Angle between directions (°)"))
        self.label_49.setText(_translate("SAXS_TOOLBOX", "Offset (°)"))
        self.label_50.setText(_translate("SAXS_TOOLBOX", "Half width (°)"))
        self.sector_qmin_label.setText(_translate("SAXS_TOOLBOX", "q min (Å⁻¹)"))
        self.sector_qmax_label.setText(_translate("SAXS_TOOLBOX", "q max (Å⁻¹)"))
        self.sector_qmin_display_lineEdit.setToolTip(_translate("SAXS_TOOLBOX", "Plot display range only \u2014 it does not change the saved .dat files. Leave empty for the full range."))
        self.sector_qmax_display_lineEdit.setToolTip(_translate("SAXS_TOOLBOX", "Plot display range only \u2014 it does not change the saved .dat files. Leave empty for the full range."))
        self.sector_regrouppushButton.setText(_translate("SAXS_TOOLBOX", "Plot"))
        self.sector_save_pushButton.setText(_translate("SAXS_TOOLBOX", "Save"))
        self.sector_save_pushButton.setToolTip(_translate("SAXS_TOOLBOX", "Write one I(q) .dat file per sector to integrations/ next to the data"))
        self.azim_q_label.setText(_translate("SAXS_TOOLBOX", "q (Å⁻¹)"))
        self.azim_threshold_label.setText(_translate("SAXS_TOOLBOX", "threshold (Å⁻¹)"))
        self.azim_threshold_doubleSpinBox.setToolTip(_translate("SAXS_TOOLBOX", "Integrate over the ring q ± threshold"))
        self.azim_plot_pushButton.setText(_translate("SAXS_TOOLBOX", "Plot"))
        self.azim_save_pushButton.setText(_translate("SAXS_TOOLBOX", "Save"))
        self.azim_save_pushButton.setToolTip(_translate("SAXS_TOOLBOX", "Write I(χ) .dat files to azimuthal_profiles/ next to the data"))
        self.sector_plot_tabWidget.setTabText(0, _translate("SAXS_TOOLBOX", "Sector regrouping I(q)"))
        self.sector_plot_tabWidget.setTabText(1, _translate("SAXS_TOOLBOX", "Azimuthal profile I(χ)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("SAXS_TOOLBOX", "Sector regrouping"))
        self.image_files_groupBox.setTitle(_translate("SAXS_TOOLBOX", "Images (sample files, grouped)"))
        self.saveSelectedImages_pushButton.setText(_translate("SAXS_TOOLBOX", "Save selected"))
        self.browseimage_pushButton_2.setText(_translate("SAXS_TOOLBOX", "Save all"))
        self.browseimage_pushButton_2.setToolTip(_translate("SAXS_TOOLBOX", "Save every image in the list as png (png_images/ next to the data)"))
        self.image_params_groupBox.setTitle(_translate("SAXS_TOOLBOX", "Image parameters"))
        self.label_10.setText(_translate("SAXS_TOOLBOX", "qmin (Å⁻¹)"))
        self.label_11.setText(_translate("SAXS_TOOLBOX", "qmax (Å⁻¹)"))
        self.label_13.setText(_translate("SAXS_TOOLBOX", "q circles [q0, q1, ...]"))
        self.label_4.setText(_translate("SAXS_TOOLBOX", "vmin (log)"))
        self.label_6.setText(_translate("SAXS_TOOLBOX", "vmax (log)"))
        self.vmin_lineEdit.setToolTip(_translate("SAXS_TOOLBOX", "Colour scale lower limit, as log10 of the normalized intensity"))
        self.vmax_lineEdit.setToolTip(_translate("SAXS_TOOLBOX", "Colour scale upper limit, as log10 of the normalized intensity"))
        self.label_8.setText(_translate("SAXS_TOOLBOX", "colormap"))
        self.cmap_comboBox.setItemText(0, _translate("SAXS_TOOLBOX", "turbo"))
        self.cmap_comboBox.setItemText(1, _translate("SAXS_TOOLBOX", "viridis"))
        self.cmap_comboBox.setItemText(2, _translate("SAXS_TOOLBOX", "jet"))
        self.cmap_comboBox.setItemText(3, _translate("SAXS_TOOLBOX", "inferno"))
        self.show_checkBox.setText(_translate("SAXS_TOOLBOX", "Show figure(s) when saving"))
        self.drawSectors_checkBox.setText(_translate("SAXS_TOOLBOX", "Draw integration sectors"))
        self.drawSectors_checkBox.setToolTip(_translate("SAXS_TOOLBOX", "Uses the directions, angle, offset and half width of the Sector regrouping tab"))
        self.checkBox.setText(_translate("SAXS_TOOLBOX", "Rotate 90°"))
        self.saveNpz_checkBox.setText(_translate("SAXS_TOOLBOX", "Save .npz"))
        self.saveNpz_checkBox.setToolTip(_translate("SAXS_TOOLBOX", "Also save the regridded Qx/Qz map as .npz next to the png, for edf tools > plot 2D image saved as npz file"))
        self.image_preview_tabWidget.setTabText(0, _translate("SAXS_TOOLBOX", "Preview"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("SAXS_TOOLBOX", "Save 2D SAXS image"))
        self.menuFile.setTitle(_translate("SAXS_TOOLBOX", "Menu"))
        self.menupyFAI.setTitle(_translate("SAXS_TOOLBOX", "pyFAI"))
        self.menuGenerate_mask.setTitle(_translate("SAXS_TOOLBOX", "Generate mask"))
        self.menuInspect_data.setTitle(_translate("SAXS_TOOLBOX", "Inspect data"))
        self.menuplot_1D_data_file_s_2.setTitle(_translate("SAXS_TOOLBOX", "plot 1D data file(s)"))
        self.menuSimulation.setTitle(_translate("SAXS_TOOLBOX", "Simulation"))
        self.menuedf_tools.setTitle(_translate("SAXS_TOOLBOX", "edf tools"))
        self.actionSet_Working_Directory.setText(_translate("SAXS_TOOLBOX", "Set Working Directory"))
        self.actionQuit.setText(_translate("SAXS_TOOLBOX", "Initialize Form"))
        self.actionInitialize_Form.setText(_translate("SAXS_TOOLBOX", "Initialize Form"))
        self.actionView_image.setText(_translate("SAXS_TOOLBOX", "View image"))
        self.actionOpen_view_image.setText(_translate("SAXS_TOOLBOX", "Open and view image"))
        self.actionDetector_calibration.setText(_translate("SAXS_TOOLBOX", "Detector calibration"))
        self.actionFrame_integration.setText(_translate("SAXS_TOOLBOX", "Frame integration"))
        self.actionSWING.setText(_translate("SAXS_TOOLBOX", "from SWING"))
        self.actionfrom_ID15.setText(_translate("SAXS_TOOLBOX", "from ID15"))
        self.actionfrom_scratch.setText(_translate("SAXS_TOOLBOX", "using an image file"))
        self.actionconvert_foxtrot_mask.setText(_translate("SAXS_TOOLBOX", "convert foxtrot mask"))
        self.actionAverage_frames.setText(_translate("SAXS_TOOLBOX", "Average frames"))
        self.actionloglog.setText(_translate("SAXS_TOOLBOX", "loglog"))
        self.actionlinear.setText(_translate("SAXS_TOOLBOX", "linear"))
        self.actionOpen_view_image_2.setText(_translate("SAXS_TOOLBOX", "Open and view image"))
        self.actionloglog_2.setText(_translate("SAXS_TOOLBOX", "log-log plot"))
        self.actionlinear_2.setText(_translate("SAXS_TOOLBOX", "linear plot"))
        self.actionAverage_frames_2.setText(_translate("SAXS_TOOLBOX", "Average frames"))
        self.actionSWING_2.setText(_translate("SAXS_TOOLBOX", "from SWING"))
        self.actionfrom_ID15_2.setText(_translate("SAXS_TOOLBOX", "from ID15"))
        self.actionfrom_ID02_SAXS.setText(_translate("SAXS_TOOLBOX", "from ID02_SAXS"))
        self.actionfrom_ID02_WAXS.setText(_translate("SAXS_TOOLBOX", "from ID02_WAXS"))
        self.actionBatch_radial_profile_plot.setText(_translate("SAXS_TOOLBOX", "Batch_radial-profile & plot"))
        self.actionCave_frames.setText(_translate("SAXS_TOOLBOX", "Cave frames"))
        self.actionCompute_I_q_from_pdb_file.setText(_translate("SAXS_TOOLBOX", "Compute I(q) from pdb file"))
        self.actionbatch_horizontal_regrouping.setText(_translate("SAXS_TOOLBOX", "batch_horizontal_regrouping"))
        self.actionbatch_vertical_regrouping.setText(_translate("SAXS_TOOLBOX", "batch_vertical_regrouping"))
        self.actionIq_from_pdb.setText(_translate("SAXS_TOOLBOX", "Calculate I(q) from pdb file"))
        self.actionGenerate_mask.setText(_translate("SAXS_TOOLBOX", "using edf file"))
        self.actionconvert_edf_to_png.setText(_translate("SAXS_TOOLBOX", "plot 2D image saved as npz file"))
        self.actionconvert_edf_to_png_2.setText(_translate("SAXS_TOOLBOX", "plot 2D image saved as npz file"))
        self.actionaverage_edf_files.setText(_translate("SAXS_TOOLBOX", "average edf files"))
        self.actionaverage_and_convert_to_png.setText(_translate("SAXS_TOOLBOX", "average and convert to png"))
        self.actioncompute_offset.setText(_translate("SAXS_TOOLBOX", "compute main orientation (offset)"))

class _SavedDataset:
    """What Save needs from a SAXSDataSet once plotted, without keeping its image arrays in memory."""

    def __init__(self, ds):
        self.label, self.folder, self.samplename = ds.label, ds.folder, ds.samplename
        self.samplefiles = types.SimpleNamespace(files=list(ds.samplefiles.files))
        self.I_0, self.k, self.calibration_coef = ds.I_0, ds.k, ds.calibration_coef

    save_profile = SAXSDataSet.save_profile


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        # Initialize ui
        self.ui = Ui_SAXS_TOOLBOX()
        self.ui.setupUi(self)

        # Intialize Menu
        self.ui.actionSet_Working_Directory.triggered.connect(self.getWD)
        self.ui.actionInitialize_Form.triggered.connect(self.init_form)
        self.path = SAXS_ROOT_DIR

        # Viewer
        self.ui.actionOpen_view_image_2.triggered.connect(self.openfile4viewer) 
        self.ui.actionloglog_2.triggered.connect(self.plotloglog)
        self.ui.actionlinear_2.triggered.connect(self.plotlinear)

        # edf tools menu
        self.ui.actionconvert_edf_to_png_2.triggered.connect(self.plot_npz)
        self.ui.actionaverage_edf_files.triggered.connect(self.average_edf_files)
        self.ui.actioncompute_offset.triggered.connect(self.compute_offset)
        # mask
        self.ui.menuGenerate_mask.triggered.connect(self.generatemask)
        
        # Initialize calibration section
        self.ui.water_calib_file_pushButton.clicked.connect(self.openfile_calib_water)
        self.ui.water_ref_calib_file_pushButton.clicked.connect(self.openfile_calib_water_ref)
        
        # Open mask
        self.ui.mask_file_pushButton_2.clicked.connect(self.openmaskfile)

        # Initialize scattering data section
        self.ui.ref_file_pushButton.clicked.connect(self.openreffile)
        self.ui.data_file_pushButton.clicked.connect(self.opendatafiles)
        self.ui.force_distance_checkBox.toggled.connect(self.ui.force_distance_lineEdit.setEnabled)

        # Radial regrouping tab
        self.ui.radial_regrouppushButton.clicked.connect(self.plot_radial)
        self.ui.radial_save_pushButton.clicked.connect(self.save_radial)

        # Sector regrouping tab
        self.ui.sector_regrouppushButton.clicked.connect(self.plot_sectors)
        self.ui.sector_save_pushButton.clicked.connect(self.save_sectors)
        self.ui.azim_plot_pushButton.clicked.connect(self.plot_azimuthal)
        self.ui.azim_save_pushButton.clicked.connect(self.save_azimuthal)

        for widget in (self.ui.MplWidget, self.ui.MplWidget_4, self.ui.MplWidget_5):
            widget.canvas.axes.clear()
            widget.canvas.figure.tight_layout()

        # 2D SAXS image tab
        self.qcircles = None
        self.ui.browseimage_pushButton_2.clicked.connect(self.batch_plot2d_vsq)
        self.ui.saveSelectedImages_pushButton.clicked.connect(self.save_selected_images)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_dataset = None
        self._preview_qmap = None
        self.ui.image_listWidget.currentItemChanged.connect(lambda *_: self._schedule_preview())
        for line_edit in (self.ui.qminplot_lineEdit, self.ui.qmaxplot_lineEdit, self.ui.qcircles_lineEdit,
                          self.ui.vmin_lineEdit, self.ui.vmax_lineEdit, self.ui.offset_lineEdit, self.ui.width_lineEdit):
            line_edit.editingFinished.connect(self._schedule_preview)
        self.ui.cmap_comboBox.currentIndexChanged.connect(lambda *_: self._schedule_preview())
        for check_box in (self.ui.drawSectors_checkBox, self.ui.checkBox):
            check_box.toggled.connect(lambda *_: self._schedule_preview())
        for spin_box in (self.ui.number_azimuthspinBox, self.ui.angle_spinBox):
            spin_box.valueChanged.connect(lambda *_: self._schedule_preview())
        self.ui.nb_samples_averaged_lineEdit.editingFinished.connect(self._refresh_image_list)
        self.ui.tabWidget.currentChanged.connect(self._on_tab_changed)
        for canvas in (self.ui.MplWidget.canvas, self.ui.MplWidget_4.canvas,
                       self.ui.MplWidget_5.canvas, self.ui.image_preview_canvas):
            canvas.mpl_connect('motion_notify_event', self._show_cursor_position)

        # Results of the last Plot per profile kind, so Save writes exactly what is shown.
        self._profile_cache = {}
        self.init_form()
        self.ui.show_checkBox.setChecked(False)
        self.ui.drawSectors_checkBox.setChecked(False)
        self.ui.saveNpz_checkBox.setChecked(False)
        self.ui.pyFAImask_checkBox.setChecked(True)
        self.ui.qminplot_lineEdit.setText("0")
        self.ui.qmaxplot_lineEdit.setText("0.2")
        self.ui.vmin_lineEdit.setText("-4")
        self.ui.vmax_lineEdit.setText("0")
    
    def init_form(self):   
        self.ui.console_textEdit.setText("")
        self.ui.lineEdit.setText('1')
        self.ui.qmin_lineEdit.setText("0.9")
        self.ui.I0_lineEdit.setText("")
        self.ui.ref_Multiplier_lineEdit.setText('1')
        self.ui.nb_samples_averaged_lineEdit.setText("1")

        self.maskpathname = []
        self.mask_autofilled = False
        self.ui.mask_file_lineEdit_2.setText("")
        self.refpathname = []
        self.ui.ref_file_lineEdit.setText("")
        self.datafile_list = []
        self.ui.data_file_textEdit.setPlainText("")
        self.water_calib_file = None
        self.ui.water_calib_file_lineEdit.setText("")
        self.water_ref_calib_file = None
        self.ui.water_ref_calib_file_lineEdit.setText("")

        self.ui.radial_azimuth_doubleSpinBox.setValue(0)
        self.ui.radial_width_doubleSpinBox.setValue(360)
        self.ui.number_azimuthspinBox.setValue(2)
        self.ui.angle_spinBox.setValue(90)
        self.ui.offset_lineEdit.setText('-90')
        self.ui.width_lineEdit.setText('10')
        self.ui.azim_q_doubleSpinBox.setValue(0.05)
        self.ui.azim_threshold_doubleSpinBox.setValue(0.0025)
        for line_edit in (self.ui.radial_qmin_display_lineEdit, self.ui.radial_qmax_display_lineEdit,
                          self.ui.sector_qmin_display_lineEdit, self.ui.sector_qmax_display_lineEdit):
            line_edit.setText("")
        self._profile_cache = {}
        self._preview_dataset = None
        self._preview_qmap = None
        self._refresh_image_list()
        return
      
    def getWD(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select the working directory', default_browse_dir(self.path))
        if folder:
            self.path = folder
        self._log(f"Working directory: {self.path}")
        return self.path 

    def _show_cursor_position(self, event):
        if event.inaxes is None:
            self.ui.statusbar.clearMessage()
        else:
            self.ui.statusbar.showMessage(event.inaxes.format_coord(event.xdata, event.ydata))

    def _log(self, text):
        self.ui.console_textEdit.moveCursor(QtGui.QTextCursor.End)
        self.ui.console_textEdit.insertPlainText(text + "\n")
        self.ui.console_textEdit.ensureCursorVisible()

    def openfile4viewer(self):
        self.view_file_path=QtWidgets.QFileDialog.getOpenFileName( self,"Open image file","%s"%self.path,'*.edf')
        self.view_file=self.view_file_path[0]
        if self.view_file!='':
            subprocess.run(['fabio_viewer',self.view_file])         
        
    
    def is_line_float(self,line):
        try:
            # Attempt to convert each element in the line to a float
            _ = [float(x) for x in line.strip().split()]
            return True
        except ValueError:
            return False
        
    def find_start_row(self,filename):
        with open(filename, 'r') as file:
            for num, line in enumerate(file, 1):
                isfloat=self.is_line_float(line)
                if isfloat:
                    return num    
    
    def plotloglog(self):
        
        filelist=QtWidgets.QFileDialog.getOpenFileNames( self,"Open 1D data file","%s"%self.path,'*.dat')
        filenames=filelist[0]
        
        plt.figure(1)
        plt.xlabel(r'Q ($\mathrm{\AA}^{-1}$ )')
        plt.ylabel(r'I')
        i=0
        for file in filenames:
            #print(file)
            label=os.path.basename(file).split('/')[-1].split('.')[0]
            index=self.find_start_row(str(file))
            q,i=np.loadtxt(str(file),usecols=(0,1),skiprows=index,unpack=True)
        
            plt.loglog(q,i,label=label)
            i+=1
        plt.legend()
        plt.tight_layout()
        plt.show()
        pass

    def plotlinear(self):
        
        filelist=QtWidgets.QFileDialog.getOpenFileNames( self,"Open 1D data file","%s"%self.path,'*.dat')
        filenames=filelist[0]
        
        plt.figure(1)
        plt.xlabel(r'Q ($\mathrm{\AA}^{-1}$ )')
        plt.ylabel(r'I (1/cm)')
        i=0
        for file in filenames:
            #print(file)
            label=os.path.basename(file).split('/')[-1].split('.')[0]
            index=self.find_start_row(str(file))
            q,i=np.loadtxt(str(file),usecols=(0,1),skiprows=index,unpack=True)
        
            plt.plot(q,i,label=label)
            i+=1
        plt.legend()
        plt.tight_layout()
        plt.show()
        pass

    def edf2png(self):
        filelist = QtWidgets.QFileDialog.getOpenFileNames(self, "Open edf files to convert", "%s" % self.path, '*.edf')
        for file in filelist[0]:
            edf_img = fabio.open(file)
            header=edf_img.header
            samplename=header['Comment']
            data = edf_img.data  # numpy array

            # nettoyage des nan
            data = np.nan_to_num(data, nan=1e-10)  # remplace les NaN par 1e-10
            data[data <= 0] = 1e-6                # évite log(0) ou log(valeurs négatives)
            data_norm = data / np.max(data)      # normalisation
            data_log = np.log10(data_norm)
            data_log = np.clip(data_log, a_min=vmin, a_max=vmax) # peut être faut-il ajuster a_min en focntion des images...?
            
            # Prepare output path
            file_number = file.split('/')[-1].split('.')[0].split('_')[-1]
            if '-' in file_number:
                file_number=file_number.split('-')[0]
            pngpath = os.path.join(self.path, 'png_images')
            os.makedirs(pngpath, exist_ok=True)
            outputfile = os.path.join(pngpath, f'{samplename}_Img{int(file_number):03d}.png')

            # Plot and save as PNG
            plt.imshow(data_log, cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title(samplename)
            #plt.legend()
            plt.savefig(outputfile, bbox_inches='tight', pad_inches=0)
            
            plt.close()


    def plot_npz(self):
        npzfile = QtWidgets.QFileDialog.getOpenFileNames(
            self, 
            "Open npz file corresponding to 2D image", 
            "%s" % self.path, 
            '*.npz'
        )
        print(npzfile[0][0])
        data = np.load(str(npzfile[0][0]))
        
        QX, QZ, Z = data["Qx"], data["QZ"], data["Z"]

        fig, ax = plt.subplots(dpi=200)

        norm = LogNorm(vmin=10**(-3), vmax=10**0)
        mesh = ax.pcolormesh(QX, QZ, Z, shading='auto', cmap='viridis', norm=norm)

        ax.set_xlabel(r"$Q_x (\AA^{-1})$", fontsize=12)
        ax.set_ylabel(r"$Q_z (\AA^{-1})$", fontsize=12)
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label("Normalized Intensity", fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        ax.set_aspect('equal')
        title=npzfile[0][0].split('/')[-1].split('.')[0] 
        ax.set_title(title,fontsize = 10)
        plt.tight_layout()

        # ---- Ajout : afficher coordonnées polaires dans la barre de statut ----
        def format_coord(x, y):
            r = np.sqrt(x**2 + y**2)
            theta = np.degrees(np.arctan2(y, x))
            return f"q={r:.4f},d={2*np.pi/(10*r):.4f}nm, θ={theta:.1f}°"

        ax.format_coord = format_coord

        plt.show()       
    
    def average_edf_files(self):
        file_list = QtWidgets.QFileDialog.getOpenFileNames(self, "Open edf files to average", "%s" % self.path, '*.edf')
        files=EdfFiles(file_list[0])
        outputname=self.path+f'/{files.samplename}_Averaged_Img_{files.number}.edf'
        files.writeedf(outputname,files.data,files.samplename)

    




    def run_sasview(self):
        import numpy as np        
        from sas.sascalc.calculator import sas_gen
        import os
        try:
            qmin=0.001
            qmax=2
            qstep=0.001
            pdbfile=QtWidgets.QFileDialog.getOpenFileName( self,"Open pdb file",self.path,'*.pdb')
            pdbfilepath=pdbfile[0]
            # load pdb file
            pdbloader = sas_gen.PDBReader()
            pdbData = pdbloader.read(str(pdbfilepath))
            
            # create genSAS class and set data
            model = sas_gen.GenSAS()
            model.set_sld_data(pdbData)
            # calculate along given q range
            q_vals = np.linspace(qmin, qmax, int(qmax/qstep))
            output = model.run([q_vals, []])
            
            iq_file=os.path.join(os.path.dirname(pdbfilepath),os.path.basename(pdbfilepath).replace(".pdb", "_sasview.iq"))
            np.savetxt(iq_file,np.column_stack((q_vals,output)),delimiter='    ',newline='\n')
            plt.figure()
            plt.loglog(q_vals,output,label=os.path.basename(pdbfilepath))
            
            plt.xlabel(r'Q (1/ $\mathrm{\AA}$ )')
            plt.ylabel(r'I (1/cm)')
            plt.legend()
            plt.show() 
            self.ui.console_textEdit.insertPlainText('I(q) data are saved in %s.\n'%str(iq_file))
        except:
            self.ui.console_textEdit.insertPlainText('An error occured.\n')
        return
    
    def generatemask(self):
        #try:
        self.file4maskpath=QtWidgets.QFileDialog.getOpenFileNames( self,"Open image file to generate a mask","%s"%self.path,'*.edf')
        self.file4maskpathname=self.file4maskpath[0]
        subprocess.run(['pyFAI-drawmask', f"{self.file4maskpathname[0]}"])
        
        #except:
        #    self.ui.console_textEdit.insertPlainText("An error occured, please try again \n")        
        return

    ## Fonctions d'ouverture de fichiers
    
    def _last_dir(self, files):
        return os.path.dirname(files[0]) if files and files[0] else self.path

    def _describe_files(self, files):
        names = find_logbook_samplenames(files)
        return [f"{os.path.basename(f)}   —   {name or '(no logbook match)'}" for f, name in zip(files, names)]

    def autofill_from_logbook(self, data_file):
        i0 = find_logbook_i0(data_file)
        if i0 is not None:
            i0_text = str(int(i0)) if i0.is_integer() else str(i0)
            self.ui.I0_lineEdit.setText(i0_text)
            self.ui.console_textEdit.insertPlainText(f"I_0={i0_text} auto-filled from logbook.csv.\n")
        else:
            self.ui.console_textEdit.insertPlainText(
                f"No logbook.csv / I_0 line next to this file — I_0 left at {self.ui.I0_lineEdit.text()}.\n")

        # Never replace a mask the user picked by hand, only an empty or auto-filled one.
        has_manual_mask = bool(self.maskpathname and self.maskpathname[0]) and not self.mask_autofilled
        if has_manual_mask:
            return
        mask = find_directory_mask(data_file)
        if mask is not None:
            self.maskpathname = [mask]
            self.mask_autofilled = True
            self.ui.mask_file_lineEdit_2.setText(os.path.basename(mask))
            self.ui.console_textEdit.insertPlainText(
                f"Mask auto-filled: {mask}\n(check that 'mask obtained from pyFAI' matches this file)\n")

    def openmaskfile(self):

        self.maskpath=QtWidgets.QFileDialog.getOpenFileNames( self,"Open pyFAI Mask File(s)",default_browse_dir(self._last_dir(self.maskpathname)),"Mask file from pyFAI *.edf" )
        if not self.maskpath[0]:
            return self.maskpathname
        self.maskpathname=self.maskpath[0]
        self.mask_autofilled = False
        #print(self.maskpathname)
        res = ''.join(os.path.basename(i) + '\n' 
                            for i in self.maskpathname)
        self.ui.mask_file_lineEdit_2.setText(res)
        
        if res!='':
            self.ui.console_textEdit.insertPlainText("Mask file successfully provided.\n")
        else:
            self.ui.console_textEdit.insertPlainText("No mask file provided, please try again\n")
           
        return self.maskpathname
    
    def openreffile(self):
        _folder, files = pick_files_by_name(self, self._last_dir(self.refpathname))
        if not files:
            self.ui.console_textEdit.insertPlainText("No reference file provided, please try again\n")
            return self.refpathname
        self.refpathname = files
        self.ui.ref_file_lineEdit.setText(' ; '.join(self._describe_files(files)))
        self.ui.console_textEdit.insertPlainText("Reference file(s) provided:\n" + '\n'.join(self._describe_files(files)) + '\n')
        return self.refpathname


    def opendatafiles (self):
        _folder, files = pick_files_by_name(self, self._last_dir(self.datafile_list))
        if not files:
            self._log("No data file provided, please try again")
            return self.datafile_list
        self.datafile_list = files
        self.ui.data_file_textEdit.setPlainText('\n'.join(self._describe_files(files)))
        self._log(f"{len(files)} sample data file(s) provided.")
        self.autofill_from_logbook(files[0])
        self._refresh_image_list()
        return self.datafile_list

    def openfile_calib_water(self):
        _folder, files = pick_files_by_name(self, self._last_dir(self.water_calib_file))
        if not files:
            self._log("No calibration file provided, please try again")
            return
        self.water_calib_file = files
        self.ui.water_calib_file_lineEdit.setText(' ; '.join(self._describe_files(files)))
        self._log("Calibration file successfully provided.")

    def openfile_calib_water_ref(self):
        _folder, files = pick_files_by_name(self, self._last_dir(self.water_ref_calib_file))
        if not files:
            self._log("No reference file provided, please try again")
            return
        self.water_ref_calib_file = files
        self.ui.water_ref_calib_file_lineEdit.setText(' ; '.join(self._describe_files(files)))
        self._log("Reference file for calibration successfully provided.")

    def create_datafile_list(self,data):
        if not data:
            raise ValueError("Select the sample data files first (Process data tab).")
        try:
            n = int(self.ui.nb_samples_averaged_lineEdit.text())
        except ValueError:
            raise ValueError("'Group & Average by bunch of' must be a whole number.")
        if n <= 0:
            raise ValueError("'Group & Average by bunch of' must be at least 1.")
        return [data[i:i+n] for i in range(0, len(data), n)]

    def _float_field(self, line_edit, name):
        try:
            return float(line_edit.text())
        except ValueError:
            raise ValueError(f"{name} must be a number (currently '{line_edit.text()}').")

    def _processing_inputs(self):
        """Everything make_SAXSDataSet reads, validated; also used as the cache key for plotted results."""
        if not self.maskpathname or not self.maskpathname[0]:
            raise ValueError("Select a mask file first (Process data tab).")
        if self.ui.water_checkBox.isChecked():
            calibrant, I_calibrant = 'water', 0.0162
        elif self.ui.hexane_checkBox.isChecked():
            calibrant, I_calibrant = 'hexane', 0.028
        else:
            calibrant, I_calibrant = None, self._float_field(self.ui.lineEdit, "Calibrant SLD")
        return dict(
            I_0=self._float_field(self.ui.I0_lineEdit, "I_0"),
            referencefiles=tuple(self.refpathname) or None,
            calibsamplefiles=tuple(self.water_calib_file) if self.water_calib_file else None,
            calibreferencefiles=tuple(self.water_ref_calib_file) if self.water_ref_calib_file else None,
            calibrant=calibrant,
            I_calibrant=I_calibrant,
            maskfile=self.maskpathname[0],
            invertmask=self.ui.pyFAImask_checkBox.isChecked(),
            k=self._float_field(self.ui.ref_Multiplier_lineEdit, "Ref. Multiplier"),
            refineflag=self.ui.ref_corr_checkBox.isChecked(),
            qseuil=self._float_field(self.ui.qmin_lineEdit, "q_min for adjusting reference subtraction"),
            force_distance=(self._optional_float_field(self.ui.force_distance_lineEdit,
                                                       "Forced sample-detector distance")
                            if self.ui.force_distance_checkBox.isChecked() else None),
        )

    def make_SAXSDataSet(self,sampledata):
        inputs = self._processing_inputs()
        for key in ('referencefiles', 'calibsamplefiles', 'calibreferencefiles'):
            if inputs[key] is not None:
                inputs[key] = list(inputs[key])
        return SAXSDataSet(sampledata, plotrefine=True, **inputs)

    def _iter_datasets(self, groups=None):
        """(file group, SAXSDataSet) for each group of data files, logging the reference refinement."""
        if groups is None:
            groups = self.create_datafile_list(self.datafile_list)
        self._processing_inputs()
        for n, sublist in enumerate(groups, 1):
            self._log(f"[{n}/{len(groups)}] {', '.join(os.path.basename(f) for f in sublist)}")
            QtWidgets.QApplication.processEvents()
            dataset = self.make_SAXSDataSet(sublist)
            if dataset.refine_info is not None:
                k, qmax_ref, qmax_sample = dataset.refine_info
                self._log(f"Refined coefficient for reference substration={k:.4f}")
                self._log(f"Solvent peak in reference measurement = {qmax_ref:.4f}")
                self._log(f"Solvent peak in sample measurement = {qmax_sample:.4f}")
                if np.abs(qmax_ref-qmax_sample) > 0.05 * qmax_ref:
                    self._log(f"**** WARNING **** \n Solvent peak positions differ by more than 5%. \n The calculated coefficient is 90% of {k/0.9:.4f}")
            yield sublist, dataset

    def save_average(self,file_list):
        files=EdfFiles(file_list)
        outputname=self.path+f'/{files.samplename}_Averaged_Img_{files.number}.edf'
        files.writeedf(outputname,files.data,files.samplename)
        return outputname

    ## Radial / sector / azimuthal profiles: Plot draws, Save writes .dat files

    def _radial_params(self):
        return dict(azimuth=self.ui.radial_azimuth_doubleSpinBox.value(),
                    width=self.ui.radial_width_doubleSpinBox.value())

    def _sector_params(self):
        return dict(nb_azimuths=self.ui.number_azimuthspinBox.value(),
                    delta=self.ui.angle_spinBox.value(),
                    offset=self._float_field(self.ui.offset_lineEdit, "Offset"),
                    width=self._float_field(self.ui.width_lineEdit, "Half width"))

    def _azimuthal_params(self):
        return dict(qvalue=self.ui.azim_q_doubleSpinBox.value(),
                    threshold=self.ui.azim_threshold_doubleSpinBox.value())

    def _radial_curves(self, ds, p):
        q, i = ds.radial_profile(p['azimuth'], p['width'])
        full_ring = p['width'] >= 360
        suffix = "" if full_ring else f"_az={p['azimuth']:g}_width={p['width']:g}"
        description = "Radial profile, whole ring" if full_ring else f"Radial profile, azimuth={p['azimuth']:g}° full width={p['width']:g}°"
        return [dict(label=ds.label, x=q, y=i, subdir='radial_profiles', filename=f"{ds.label}_I(q){suffix}.dat",
                     columns="q (A^-1)    I", description=description)]

    def _sector_curves(self, ds, p):
        # One line style per direction, so the sectors stay distinguishable
        # when every sample also has its own colour.
        return [dict(label=f"{ds.label} az={az:g}°", x=q, y=i, subdir='integrations',
                     linestyle=SECTOR_LINESTYLES[n % len(SECTOR_LINESTYLES)],
                     filename=f"{ds.label}_I(q)_az={int(az)}°_fullwidth={2*p['width']:g}.dat",
                     columns="q (A^-1)    I", description=f"Sector profile, azimuth={az:g}° half width={p['width']:g}°")
                for n, (az, q, i) in enumerate(ds.sector_profiles(**p))]

    def _azimuthal_curves(self, ds, p):
        chi, i = ds.azimuthal_profile(p['qvalue'], p['threshold'])
        return [dict(label=ds.label, x=chi, y=i, subdir='azimuthal_profiles',
                     filename=f"{ds.label}_I(chi)_q={p['qvalue']:.4f}±{p['threshold']:.4f}.dat",
                     columns="chi (deg)    I", description=f"Azimuthal profile, q={p['qvalue']:g} ± {p['threshold']:g} A^-1")]

    def _profile_spec(self, kind):
        return {
            'radial': dict(params=self._radial_params, curves=self._radial_curves, widget=self.ui.MplWidget,
                           xlabel='Q ($\\AA^{-1}$)', loglog=True, page=None, format_coord=format_q_coord,
                           display_range=lambda: (self.ui.radial_qmin_display_lineEdit,
                                                  self.ui.radial_qmax_display_lineEdit),
                           title=lambda p: "" if p['width'] >= 360 else f"azimuth {p['azimuth']:g}°, full width {p['width']:g}°"),
            'sector': dict(params=self._sector_params, curves=self._sector_curves, widget=self.ui.MplWidget_4,
                           xlabel='Q ($\\AA^{-1}$)', loglog=True, page=0, format_coord=format_q_coord,
                           display_range=lambda: (self.ui.sector_qmin_display_lineEdit,
                                                  self.ui.sector_qmax_display_lineEdit),
                           title=lambda p: ""),
            'azimuthal': dict(params=self._azimuthal_params, curves=self._azimuthal_curves, widget=self.ui.MplWidget_5,
                              xlabel='$\\chi$ (°)', loglog=False, page=1, format_coord=format_chi_coord,
                              display_range=None,
                              title=lambda p: f"q = {p['qvalue']:g} ± {p['threshold']:g} $\\AA^{{-1}}$"),
        }[kind]

    def _optional_float_field(self, line_edit, name):
        text = line_edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{name} must be a number or left empty (currently '{line_edit.text()}').")

    def _apply_display_range(self, ax, spec):
        """Zoom the axes to the display q range. Only the view changes; saved files keep every point."""
        if spec['display_range'] is None:
            return
        qmin_edit, qmax_edit = spec['display_range']()
        qmin = self._optional_float_field(qmin_edit, "display q min")
        qmax = self._optional_float_field(qmax_edit, "display q max")
        if qmin is None and qmax is None:
            return
        ax.set_xlim(left=qmin, right=qmax)
        # Rescale the intensity axis to the visible q range, otherwise the
        # curve is squashed by points that are no longer shown.
        left, right = ax.get_xlim()
        visible = []
        for line in ax.get_lines():
            x, y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
            keep = (x >= left) & (x <= right) & np.isfinite(y)
            if spec['loglog']:
                keep &= y > 0
            visible.append(y[keep])
        visible = np.concatenate(visible) if visible else np.array([])
        if visible.size:
            low, high = visible.min(), visible.max()
            if spec['loglog']:
                ax.set_ylim(low / 1.5, high * 1.5)
            else:
                margin = 0.05 * (high - low) or 1.0
                ax.set_ylim(low - margin, high + margin)

    def _profile_key(self, kind, params):
        groups = self.create_datafile_list(self.datafile_list)
        inputs = self._processing_inputs()
        return (tuple(map(tuple, groups)), tuple(sorted(inputs.items())), tuple(sorted(params.items())))

    def _compute_profiles(self, kind, plot):
        spec = self._profile_spec(kind)
        params = spec['params']()
        key = self._profile_key(kind, params)
        if plot:
            if spec['page'] is not None:
                self.ui.sector_plot_tabWidget.setCurrentIndex(spec['page'])
            fig, ax = spec['widget'].canvas.figure, spec['widget'].canvas.axes
            ax.clear()
            ax.set_xlabel(spec['xlabel'])
            ax.set_title(spec['title'](params), fontsize=10)
            ax.format_coord = spec['format_coord']
        results = []
        groups = self.create_datafile_list(self.datafile_list)
        colors = sample_colors(len(groups))
        for n, (sublist, ds) in enumerate(self._iter_datasets(groups)):
            curves = spec['curves'](ds, params)
            results.append((sublist, ds, curves))
            if plot:
                ax.set_ylabel('Intensity $(cm^{-1})$' if ds.absolute else 'Intensity')
                for curve in curves:
                    # One colour per sample, one line style per sector direction.
                    plot_line = ax.loglog if spec['loglog'] else ax.plot
                    plot_line(curve['x'], curve['y'], label=curve['label'],
                              color=colors[n], linestyle=curve.get('linestyle', '-'))
                self._apply_display_range(ax, spec)
                place_legend_outside(fig, ax)
                fig.canvas.draw()
                fig.canvas.flush_events()
                QtWidgets.QApplication.processEvents()
        # Only the curves and dataset metadata are kept, not the (large) image arrays.
        self._profile_cache[kind] = (key, [(sublist, _SavedDataset(ds), curves) for sublist, ds, curves in results])
        self._log(f"{kind.capitalize()} profiles computed for {len(results)} group(s).")
        return self._profile_cache[kind][1]

    def _plot_profiles(self, kind):
        self._compute_profiles(kind, plot=True)

    def _save_profiles(self, kind):
        spec = self._profile_spec(kind)
        cached = self._profile_cache.get(kind)
        if cached is not None and cached[0] == self._profile_key(kind, spec['params']()):
            results = cached[1]
            self._log("Saving the plotted profiles.")
        else:
            self._log("Parameters changed since the last Plot (or nothing plotted yet): recomputing before saving.")
            results = self._compute_profiles(kind, plot=False)
        written = 0
        for sublist, ds, curves in results:
            for curve in curves:
                path = ds.save_profile(curve['subdir'], curve['filename'], curve['x'], curve['y'],
                                       curve['columns'], curve['description'])
                written += 1
            if self.ui.save_avge_edf_checkBox.isChecked():
                self._log(f"Averaged edf saved: {self.save_average(sublist)}")
        self._log(f"{written} file(s) saved, e.g. {path}" if written else "Nothing to save.")

    @report_errors
    def plot_radial(self):
        self._plot_profiles('radial')

    @report_errors
    def save_radial(self):
        self._save_profiles('radial')

    @report_errors
    def plot_sectors(self):
        self._plot_profiles('sector')

    @report_errors
    def save_sectors(self):
        self._save_profiles('sector')

    @report_errors
    def plot_azimuthal(self):
        self._plot_profiles('azimuthal')

    @report_errors
    def save_azimuthal(self):
        self._save_profiles('azimuthal')

    ## 2D SAXS image tab: image list, live preview, png export

    def _image_params(self):
        qcircles = self.ui.qcircles_lineEdit.text().strip()
        if qcircles:
            try:
                value = ast.literal_eval(qcircles)
                qcircles = tuple(float(q) for q in (value if isinstance(value, (list, tuple)) else [value]))
            except (ValueError, SyntaxError, TypeError):
                raise ValueError(f"q circles must be a number or a list like [0.05, 0.1] (currently '{qcircles}').")
        else:
            qcircles = None
        return dict(
            qmin=self._float_field(self.ui.qminplot_lineEdit, "qmin"),
            qmax=self._float_field(self.ui.qmaxplot_lineEdit, "qmax"),
            vmin=self._float_field(self.ui.vmin_lineEdit, "vmin"),
            vmax=self._float_field(self.ui.vmax_lineEdit, "vmax"),
            qcircles=qcircles,
            cmap=self.ui.cmap_comboBox.currentText(),
            rotate=self.ui.checkBox.isChecked(),
            draw_sectors=self.ui.drawSectors_checkBox.isChecked(),
            **self._sector_params(),
        )

    def _refresh_image_list(self):
        list_widget = self.ui.image_listWidget
        current = list_widget.currentItem().data(QtCore.Qt.UserRole) if list_widget.currentItem() else None
        list_widget.blockSignals(True)
        list_widget.clear()
        try:
            groups = self.create_datafile_list(self.datafile_list)
        except ValueError:
            groups = []
        for group, name in zip(groups, find_logbook_samplenames([g[0] for g in groups])):
            text = f"{name or '(no logbook match)'}\n    {os.path.basename(group[0])}"
            if len(group) > 1:
                text += f"  (+{len(group) - 1} averaged)"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, list(group))
            item.setToolTip("\n".join(group))
            list_widget.addItem(item)
            if list(group) == current:
                list_widget.setCurrentItem(item)
        if list_widget.currentItem() is None and list_widget.count():
            list_widget.setCurrentRow(0)
        list_widget.blockSignals(False)
        self._schedule_preview()

    def _on_tab_changed(self, _index):
        if self.ui.tabWidget.currentWidget() is self.ui.tab_2:
            self._refresh_image_list()

    def _schedule_preview(self):
        if self.ui.tabWidget.currentWidget() is self.ui.tab_2:
            self._preview_timer.start()

    def _preview_qmap_for(self, group, params):
        """Dataset and regridded map of the previewed image, rebuilt only when their inputs change."""
        dataset_key = (tuple(group), tuple(sorted(self._processing_inputs().items())))
        if self._preview_dataset is None or self._preview_dataset[0] != dataset_key:
            self._preview_dataset = (dataset_key, self.make_SAXSDataSet(list(group)))
            self._preview_qmap = None
        ds = self._preview_dataset[1]
        qmap_key = (dataset_key, params['qmin'], params['qmax'], params['rotate'])
        if self._preview_qmap is None or self._preview_qmap[0] != qmap_key:
            self._preview_qmap = (qmap_key, ds.qmap(params['qmin'], params['qmax'], params['rotate']))
        return ds, self._preview_qmap[1]

    def _update_preview(self):
        canvas = self.ui.image_preview_canvas
        fig = canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)
        item = self.ui.image_listWidget.currentItem()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if item is None:
                raise ValueError("Select sample data files in the Process data tab, then pick an image in the list.")
            params = self._image_params()
            _ds, qmap = self._preview_qmap_for(item.data(QtCore.Qt.UserRole), params)
            SAXSDataSet.draw_qmap(fig, ax, *qmap, qmax=params['qmax'], vmin=params['vmin'], vmax=params['vmax'],
                                  cmap=params['cmap'], qcircles=params['qcircles'], draw_sectors=params['draw_sectors'],
                                  nb_azimuths=params['nb_azimuths'], width=params['width'],
                                  offset=params['offset'], delta=params['delta'], fontsize=9)
            ax.set_title(item.text().split('\n')[0], fontsize=10)
            fig.tight_layout()
        except Exception as exc:
            fig.clf()
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, f"No preview:\n{exc}", ha='center', va='center', wrap=True, fontsize=10, color='dimgray')
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        canvas.draw_idle()

    def _save_images(self, groups):
        params = self._image_params()
        show = self.ui.show_checkBox.isChecked()
        saved = 0
        for sublist, dataset in self._iter_datasets(groups):
            qmap = None
            if self._preview_qmap is not None and self._preview_qmap[0] == (
                    (tuple(sublist), tuple(sorted(self._processing_inputs().items()))),
                    params['qmin'], params['qmax'], params['rotate']):
                qmap = self._preview_qmap[1]
            figname = dataset.plot2d_vsq(
                qmin=params['qmin'], qmax=params['qmax'], vmin=params['vmin'], vmax=params['vmax'],
                qcircles=params['qcircles'], cmap=params['cmap'], show=show, rotate=params['rotate'],
                draw_sectors=params['draw_sectors'], nb_azimuths=params['nb_azimuths'], delta=params['delta'],
                offset=params['offset'], width=params['width'],
                save_npz=self.ui.saveNpz_checkBox.isChecked(), qmap_result=qmap)
            self._log(f'SAXS image generated in {figname}')
            saved += 1
        self._log(f"{saved} image(s) saved.")

    @report_errors
    def batch_plot2d_vsq(self):
        self._save_images(None)

    @report_errors
    def save_selected_images(self):
        items = sorted(self.ui.image_listWidget.selectedItems(), key=self.ui.image_listWidget.row)
        if not items:
            raise ValueError("Select one or more images in the list first.")
        self._save_images([item.data(QtCore.Qt.UserRole) for item in items])

    @report_errors
    def compute_offset(self,
                       q0=0.01, dq=0.005, npt_rad=1440, npt_azim=1440,
                       plot_profile=False, plot_image=False):
        """
        Trouve l'angle azimutal maximal à q fixé et trace la droite sur l'image SAXS.

        Parameters
        ----------
        q0 : float
            Valeur de q (Å⁻¹) autour de laquelle extraire le profil.
        dq : float
            Largeur de la fenêtre radiale (Å⁻¹).
        npt_rad : int
            Nombre de bins radiaux pour l'intégration 2D.
        npt_azim : int
            Nombre de bins azimutaux.
        plot_profile : bool
            Si True, trace le profil I(χ) et marque le maximum.
        plot_image : bool
            Si True, trace la droite correspondant à l'angle maximal sur l'image.

        Returns
        -------
        max_angle : float
            Angle azimutal (°) où l’intensité est maximale à q0. Valeur comprise entre -45° et +45°
            
        
        """
       
        # regroup sample data by bunch of positions (will average)
        shaped_datalist=self.create_datafile_list(self.datafile_list)

        for sublist in shaped_datalist:
            dataset=self.make_SAXSDataSet(sublist)

            # get data from image
            data = dataset.sampledata

            ny, nx = data.shape
            ai = dataset.ai_from_edf()
            # Intégration 2D
            Iqchi, q, chi = ai.integrate2d(
                data,
                npt_azim,
                npt_rad,
                unit=("q_A^-1", "chi_deg"),
                method="bbox"
            )

            # Sélection de la plage q0 ± dq
            mask_q = (q >= q0 - dq) & (q <= q0 + dq)
            if not np.any(mask_q):
                raise ValueError(f"Aucune valeur de q trouvée dans [{q0-dq}, {q0+dq}]")
            #print(len(mask_q),Iqchi.shape[0])
            Iq = np.mean(Iqchi[mask_q, :], axis=0)
            Ichi = np.mean(Iqchi, axis =1)

            # Trouver le maximum
            max_idx = np.argmax(Ichi)
            max_angle = chi[max_idx] %90
            if max_angle >45:
                max_angle -=90
            self.ui.console_textEdit.insertPlainText(f"Determined offset is ={max_angle:.4f} \n")
            self.ui.offset_lineEdit.setText(f'{max_angle-90}')
            return max_angle        
        
def on_draw_sectors_toggled(self, checked: bool):
    # simple feedback: enable/disable the apply button or change a label
    self.ui.applySectorParams_pushButton.setEnabled(checked)

def on_apply_sector_params(self):
    # read UI fields and store them centrally so both plotting and integration use them
    try:
        nb = int(self.ui.number_azimuthspinBox.value())
        delta = float(self.ui.angle_spinBox.value())
        offset = float(self.ui.offset_lineEdit.text())
        width = float(self.ui.width_lineEdit.text())
    except Exception as e:
        QtWidgets.QMessageBox.warning(self, "Invalid sector params", str(e))
        return

    self._last_plot2d_sector_params = dict(
        nb_azimuths=nb,
        width=width,
        offset=offset,
        delta=delta
    )
    QtWidgets.QMessageBox.information(self, "Sector params", "Sector parameters stored.")        


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    QtCore.QTimer.singleShot(0, application.getWD)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
