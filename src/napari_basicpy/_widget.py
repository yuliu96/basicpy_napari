"""
TODO
[ ] Add Autosegment feature when checkbox is marked
[ ] Add text instructions to "Hover input field for tooltip"
"""

import enum
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import importlib.metadata

import numpy as np
from basicpy import BaSiC
from magicgui.widgets import create_widget
from napari.qt import thread_worker
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QDoubleValidator, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QSlider,
    QSizePolicy,
    QLineEdit,
)

if TYPE_CHECKING:
    import napari  # pragma: no cover

from magicgui.widgets import ComboBox
from napari.layers import Image

SHOW_LOGO = True  # Show or hide the BaSiC logo in the widget

logger = logging.getLogger(__name__)


def build_general_settings_containers():
    skip = ["resize_mode", "resize_params", "working_size"]
    simple_settings = ["get_darkfield"]

    def build_widget(k):
        field = BaSiC.model_fields[k]
        description = field.description
        default = field.default
        annotation = field.annotation
        # Handle enumerated settings
        try:
            if issubclass(annotation, enum.Enum):
                try:
                    default = annotation[default]
                except KeyError:
                    default = default
        except TypeError:
            pass
        # Define when to use scientific notation spinbox based on default value
        if (type(default) == float or type(default) == int) and (default < 0.01 or default > 999):
            widget = ScientificDoubleSpinBox()
            widget.native.setValue(default)
            widget.native.adjustSize()
        else:
            widget = create_widget(
                value=default,
                annotation=annotation,
                options={"tooltip": description},
            )
        widget.native.setFixedWidth(150)
        return widget

    # All settings here will be used to initialize BaSiC
    self._settings = {k: build_widget(k) for k in BaSiC().settings.keys() if k not in skip}

    # build simple settings container
    simple_settings_gb = QGroupBox("Parameters")  # make groupbox
    simple_settings_gb.setLayout(QVBoxLayout())
    simple_settings_form = QWidget()  # make form
    simple_settings_form.setLayout(QFormLayout())
    simple_settings_form.layout().setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    simple_settings_gb.layout().addWidget(simple_settings_form)  # add form to groupbox
    simple_settings_gb.setAlignment(Qt.AlignTop)
    # create advanaced settings groupbox
    advanced_settings_gb = QGroupBox("Advanced Settings")  # make groupbox
    advanced_settings_gb.setLayout(QVBoxLayout())
    advanced_settings_form = QWidget()  # make form
    advanced_settings_form.setLayout(QFormLayout())
    advanced_settings_form.layout().setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    # sort settings into either simple or advanced settings containers
    for k, v in self._settings.items():
        if k in simple_settings:
            simple_settings_form.layout().addRow(k, v.native)
        else:
            advanced_settings_form.layout().addRow(k, v.native)

    # The scroll view is created after filling the list to make it the correct size
    advanced_settings_scroll = QScrollArea()  # make scroll view
    advanced_settings_scroll.setWidget(advanced_settings_form)  # apply scroll view to form
    advanced_settings_gb.layout().addWidget(advanced_settings_scroll)  # add view to groupbox

    return simple_settings_gb, advanced_settings_gb


class GeneralSetting(QGroupBox):
    # (15.11.2024) Function 1
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setStyleSheet("QGroupBox { " "border-radius: 10px}")
        self.viewer = parent.viewer
        self.parent = parent
        self.name = ""  # layer.name

        # layout and parameters for intensity normalization
        vbox = QGridLayout()
        self.setLayout(vbox)

        skip = [
            "resize_mode",
            "resize_params",
            "working_size",
            "fitting_mode",
            "fitting_mode",
            "get_darkfield",
            "smoothness_flatfield",
            "smoothness_darkfield",
            "sparse_cost_darkfield",
            "sort_intensity",
            "device",
        ]
        _settings = {k: self.build_widget(k) for k in BaSiC().settings.keys() if k not in skip}

        # sort settings into either simple or advanced settings containers
        # _settings = {**{"device": ComboBox(choices=["cpu", "cuda"])}, **_settings}
        i = 0
        for k, v in _settings.items():
            vbox.addWidget(QLabel(k), i, 0, 1, 1)
            vbox.addWidget(v.native, i, 1, 1, 1)
            i += 1

    def build_widget(self, k):
        field = BaSiC.model_fields[k]
        description = field.description
        default = field.default
        annotation = field.annotation
        # Handle enumerated settings
        try:
            if issubclass(annotation, enum.Enum):
                try:
                    default = annotation[default]
                except KeyError:
                    default = default
        except TypeError:
            pass
        # Define when to use scientific notation spinbox based on default value
        if (type(default) == float or type(default) == int) and (default < 0.01 or default > 999):
            widget = ScientificDoubleSpinBox()
            widget.native.setValue(default)
            widget.native.adjustSize()
        else:
            widget = create_widget(
                value=default,
                annotation=annotation,
                options={"tooltip": description},
            )
        return widget


class AutotuneSetting(QGroupBox):
    # (15.11.2024) Function 1
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setStyleSheet("QGroupBox { " "border-radius: 10px}")
        self.viewer = parent.viewer
        self.parent = parent
        self.name = ""  # layer.name

        # layout and parameters for intensity normalization
        vbox = QGridLayout()
        self.setLayout(vbox)

        args = [
            "histogram_qmin",
            "histogram_qmax",
            "vmin_factor",
            "vrange_factor",
            "histogram_bins",
            "histogram_use_fitting_weight",
            "fourier_l0_norm_image_threshold",
            "fourier_l0_norm_fourier_radius",
            "fourier_l0_norm_threshold",
            "fourier_l0_norm_cost_coef",
        ]

        _default = {
            "histogram_qmin": 0.01,
            "histogram_qmax": 0.99,
            "vmin_factor": 0.6,
            "vrange_factor": 1.5,
            "histogram_bins": 1000,
            "histogram_use_fitting_weight": True,
            "fourier_l0_norm_image_threshold": 0.1,
            "fourier_l0_norm_fourier_radius": 10,
            "fourier_l0_norm_threshold": 0.0,
            "fourier_l0_norm_cost_coef": 30,
        }

        _settings = {k: self.build_widget(k, _default[k]) for k in args}
        # sort settings into either simple or advanced settings containers
        # _settings = {**{"device": ComboBox(choices=["cpu", "cuda"])}, **_settings}
        i = 0
        for k, v in _settings.items():
            vbox.addWidget(QLabel(k), i, 0, 1, 1)
            vbox.addWidget(v.native, i, 1, 1, 1)
            i += 1

    def build_widget(self, k, default):
        # Handle enumerated settings
        annotation = type(default)
        try:
            if issubclass(annotation, enum.Enum):
                try:
                    default = annotation[default]
                except KeyError:
                    default = default
        except TypeError:
            pass

        # Define when to use scientific notation spinbox based on default value
        if (type(default) == float or type(default) == int) and (default < 0.01 or default > 999):
            widget = ScientificDoubleSpinBox()
            widget.native.setValue(default)
            widget.native.adjustSize()
        else:
            widget = create_widget(
                value=default,
                annotation=annotation,
            )
        # widget.native.setMinimumWidth(150)
        return widget


class BasicWidget(QWidget):
    """Example widget class."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Init example widget."""  # noqa DAR101
        super().__init__()

        self.viewer = viewer

        # Define builder functions
        widget = QWidget()
        main_layout = QGridLayout()
        widget.setLayout(main_layout)

        # Define builder functions
        def build_header_container():
            """Build the widget header."""
            header_container = QWidget()
            header_layout = QVBoxLayout()
            header_container.setLayout(header_layout)
            # show/hide logo
            if SHOW_LOGO:
                logo_path = str((Path(__file__).parent / "_icons/logo.png").absolute())
                logo_pm = QPixmap(logo_path)
                logo_lbl = QLabel()
                logo_lbl.setPixmap(logo_pm)
                logo_lbl.setAlignment(Qt.AlignCenter)
                header_layout.addWidget(logo_lbl)
            # Show label and package version of BaSiCPy
            lbl = QLabel(f"<b>BaSiCPy Shading Correction</b>")
            lbl.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(lbl)

            return header_container

        def build_doc_reference_label():
            doc_reference_label = QLabel()
            doc_reference_label.setOpenExternalLinks(True)
            # doc_reference_label.setText(
            #     '<a href="https://basicpy.readthedocs.io/en/latest/api.html#basicpy.basicpy.BaSiC">'
            #     "See docs for settings details</a>"
            # )

            doc_reference_label.setText(
                '<a style= color:white; style= background-color:green; href="https://basicpy.readthedocs.io/en/latest/api.html#basicpy.basicpy.BaSiC">See docs for settings details</a>'
            )

            return doc_reference_label

        # Build fit widget components
        header_container = build_header_container()
        doc_reference_lbl = build_doc_reference_label()
        self.fit_widget = self.build_fit_widget_container()
        self.transform_widget = self.build_transform_widget_container()

        # Add containers/widgets to layout

        self.btn_fit = QPushButton("Fit BaSiCPy")
        self.btn_fit.setCheckable(True)
        self.btn_fit.clicked.connect(self.toggle_fit)
        self.btn_fit.setStyleSheet("""QPushButton{background:green;border-radius:5px;}""")
        self.btn_fit.setFixedWidth(400)

        self.btn_transform = QPushButton("Apply BaSiCPy")
        self.btn_transform.setCheckable(True)
        self.btn_transform.clicked.connect(self.toggle_transform)
        self.btn_transform.setStyleSheet("""QPushButton{background:green;border-radius:5px;}""")
        self.btn_transform.setFixedWidth(400)

        main_layout.addWidget(header_container, 0, 0, 1, 2)
        main_layout.addWidget(self.btn_fit, 1, 0)
        main_layout.addWidget(self.fit_widget, 2, 0)
        main_layout.addWidget(self.btn_transform, 3, 0)
        main_layout.addWidget(self.transform_widget, 4, 0)
        main_layout.addWidget(doc_reference_lbl, 6, 0)

        main_layout.setAlignment(Qt.AlignTop)

        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

        # self.run_btn.clicked.connect(self._run)

    def build_transform_widget_container(self):
        settings_container = QGroupBox("Parameters")  # make groupbox
        settings_layout = QGridLayout()
        label_timelapse = QLabel("is_timelapse:")
        label_timelapse.setFixedWidth(150)
        self.checkbox_is_timelapse = QCheckBox()
        self.checkbox_is_timelapse.setChecked(False)

        settings_layout.addWidget(label_timelapse, 0, 0)
        settings_layout.addWidget(self.checkbox_is_timelapse, 0, 1)

        settings_layout.setAlignment(Qt.AlignTop)
        settings_container.setLayout(settings_layout)

        inputs_container = self.build_transform_inputs_containers()

        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save")

        transform_layout = QGridLayout()
        transform_layout.addWidget(inputs_container, 0, 0, 1, 2)
        transform_layout.addWidget(settings_container, 1, 0, 1, 2)
        transform_layout.addWidget(self.run_btn, 2, 0, 1, 1)
        transform_layout.addWidget(self.cancel_btn, 2, 1, 1, 1)
        transform_layout.addWidget(self.save_btn, 3, 0, 1, 2)
        transform_layout.setAlignment(Qt.AlignTop)

        transform_widget = QWidget()
        transform_widget.setLayout(transform_layout)
        transform_widget.setVisible(False)

        return transform_widget

    def build_fit_widget_container(self):

        settings_container = self.build_settings_containers()
        inputs_container = self.build_inputs_containers()

        advanced_parameters = QGroupBox("Advanced parameters")
        advanced_parameters_layout = QGridLayout()
        advanced_parameters.setLayout(advanced_parameters_layout)

        # general settings
        self.general_settings = GeneralSetting(self)
        self.btn_general_settings = QPushButton("General settings")
        self.btn_general_settings.setCheckable(True)
        self.btn_general_settings.clicked.connect(self.toggle_general_settings)
        self.checkbox_get_darkfield.clicked.connect(self.toggle_lineedit_smoothness_darkfield)
        advanced_parameters_layout.addWidget(self.btn_general_settings)

        advanced_parameters_layout.addWidget(self.general_settings)

        # autotune settings
        self.autotune_settings = AutotuneSetting(self)
        self.btn_autotune_settings = QPushButton("Autotune settings")
        self.btn_autotune_settings.setCheckable(True)
        self.btn_autotune_settings.clicked.connect(self.toggle_autotune_settings)
        advanced_parameters_layout.addWidget(self.btn_autotune_settings)
        advanced_parameters_layout.addWidget(self.autotune_settings)

        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save")

        fit_layout = QGridLayout()
        fit_layout.addWidget(inputs_container, 0, 0, 1, 2)
        fit_layout.addWidget(settings_container, 1, 0, 1, 2)
        fit_layout.addWidget(advanced_parameters, 2, 0, 1, 2)
        fit_layout.addWidget(self.run_btn, 3, 0, 1, 1)
        fit_layout.addWidget(self.cancel_btn, 3, 1, 1, 1)
        fit_layout.addWidget(self.save_btn, 4, 0, 1, 2)
        fit_layout.setAlignment(Qt.AlignTop)
        fit_widget = QWidget()
        fit_widget.setLayout(fit_layout)
        fit_widget.setVisible(False)
        return fit_widget

    def build_transform_inputs_containers(self):
        input_gb = QGroupBox("Inputs")
        gb_layout = QGridLayout()

        label_image = QLabel("images:")
        label_image.setFixedWidth(150)
        label_flatfield = QLabel("flatfield:")
        label_flatfield.setFixedWidth(150)
        label_darkfield = QLabel("darkfield:")
        label_darkfield.setFixedWidth(150)

        self.image_select = ComboBox(choices=self.layers_image)
        self.flatfield_select = ComboBox(choices=self.layers_image)
        self.darkfield_select = ComboBox(choices=self.layers_weight)

        gb_layout.addWidget(label_image, 0, 0, 1, 1)
        gb_layout.addWidget(self.image_select.native, 0, 1, 1, 2)
        gb_layout.addWidget(label_flatfield, 1, 0, 1, 1)
        gb_layout.addWidget(self.flatfield_select.native, 1, 1, 1, 2)
        gb_layout.addWidget(label_darkfield, 2, 0, 1, 1)
        gb_layout.addWidget(self.darkfield_select.native, 2, 1, 1, 2)

        gb_layout.setAlignment(Qt.AlignTop)
        input_gb.setLayout(gb_layout)

        return input_gb

    def build_inputs_containers(self):
        input_gb = QGroupBox("Inputs")
        gb_layout = QGridLayout()

        label_image = QLabel("images:")
        label_image.setFixedWidth(150)
        label_fitting_weight = QLabel("segmentation mask:")
        label_fitting_weight.setFixedWidth(150)

        self.image_select = ComboBox(choices=self.layers_image)
        self.weight_select = ComboBox(choices=self.layers_weight)

        gb_layout.addWidget(label_image, 0, 0, 1, 1)
        gb_layout.addWidget(self.image_select.native, 0, 1, 1, 2)
        gb_layout.addWidget(label_fitting_weight, 1, 0, 1, 1)
        gb_layout.addWidget(self.weight_select.native, 1, 1, 1, 2)

        gb_layout.setAlignment(Qt.AlignTop)
        input_gb.setLayout(gb_layout)

        return input_gb

    def build_settings_containers(self):
        simple_settings_gb = QGroupBox("Parameters")  # make groupbox
        gb_layout = QGridLayout()

        label_get_darkfield = QLabel("get_darkfield:")
        label_timelapse = QLabel("is_timelapse:")
        label_sorting = QLabel("sort_intensity:")
        label_smoothness_flatfield = QLabel("smoothness_flatfield:")
        label_smoothness_darkfield = QLabel("smoothness_darkfield:")

        label_get_darkfield.setFixedWidth(150)
        label_timelapse.setFixedWidth(150)
        label_sorting.setFixedWidth(150)
        label_smoothness_flatfield.setFixedWidth(150)
        label_smoothness_darkfield.setFixedWidth(150)

        self.lineedit_smoothness_flatfield = QLineEdit()
        self.lineedit_smoothness_darkfield = QLineEdit()
        self.lineedit_smoothness_darkfield.setEnabled(False)
        self.lineedit_smoothness_darkfield.setText("Not available")

        self.autotune_btn = QPushButton("autotune")
        self.autotune_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.checkbox_get_darkfield = QCheckBox()
        self.checkbox_get_darkfield.setChecked(False)
        self.checkbox_is_timelapse = QCheckBox()
        self.checkbox_is_timelapse.setChecked(False)
        self.checkbox_sorting = QCheckBox()
        self.checkbox_sorting.setChecked(False)

        gb_layout.addWidget(label_get_darkfield, 0, 0)
        gb_layout.addWidget(self.checkbox_get_darkfield, 0, 1)
        gb_layout.addWidget(label_timelapse, 1, 0)
        gb_layout.addWidget(self.checkbox_is_timelapse, 1, 1)
        gb_layout.addWidget(label_sorting, 2, 0)
        gb_layout.addWidget(self.checkbox_sorting, 2, 1)
        gb_layout.addWidget(label_smoothness_flatfield, 3, 0, 1, 1)
        gb_layout.addWidget(self.lineedit_smoothness_flatfield, 3, 1, 1, 1)
        gb_layout.addWidget(label_smoothness_darkfield, 4, 0, 1, 1)
        gb_layout.addWidget(self.lineedit_smoothness_darkfield, 4, 1, 1, 1)
        gb_layout.addWidget(self.autotune_btn, 3, 2, 2, 1)

        gb_layout.setAlignment(Qt.AlignTop)
        simple_settings_gb.setLayout(gb_layout)

        return simple_settings_gb

    def toggle_lineedit_smoothness_darkfield(self, checked: bool):
        if self.checkbox_get_darkfield.isChecked():
            self.lineedit_smoothness_darkfield.setEnabled(True)
            self.lineedit_smoothness_darkfield.clear()
        else:
            self.lineedit_smoothness_darkfield.setEnabled(False)
            self.lineedit_smoothness_darkfield.setText("Not available")

    def toggle_transform(self, checked: bool):
        # Switching the visibility of the transform_widget
        if self.transform_widget.isVisible():
            self.transform_widget.setVisible(False)
        else:
            self.transform_widget.setVisible(True)
            self.fit_widget.setVisible(False)

    def toggle_fit(self, checked: bool):
        # Switching the visibility of the fit_widget
        if self.fit_widget.isVisible():
            self.fit_widget.setVisible(False)
        else:
            self.fit_widget.setVisible(True)
            self.transform_widget.setVisible(False)

    def toggle_general_settings(self, checked: bool):
        # Switching the visibility of the General settings
        if self.general_settings.isVisible():
            self.general_settings.setVisible(False)
            self.btn_general_settings.setText("General settings")
        else:
            self.general_settings.setVisible(True)
            self.btn_general_settings.setText("Hide general settings")

    def toggle_autotune_settings(self, checked: bool):
        # Switching the visibility of the Autotune settings
        if self.autotune_settings.isVisible():
            self.autotune_settings.setVisible(False)
            self.btn_autotune_settings.setText("Autotune settings")
        else:
            self.autotune_settings.setVisible(True)
            self.btn_autotune_settings.setText("Hide autotune settings")

    def layers_image(
        self,
        wdg: ComboBox,
    ) -> list[Image]:
        return ["--select input images--"] + [layer for layer in self.viewer.layers]

    def layers_weight(
        self,
        wdg: ComboBox,
    ) -> list[Image]:
        return ["none"] + [layer for layer in self.viewer.layers]

    @property
    def settings(self):
        """Get settings for BaSiC."""
        return {k: v.value for k, v in self._settings.items()}

    def _run(self):
        # disable run button
        self.run_btn.setDisabled(True)
        # get layer information
        data, meta, _ = self.layer_select.value.as_layer_data_tuple()

        # define function to update napari viewer
        def update_layer(update):
            data, flatfield, darkfield, meta = update
            self.viewer.add_image(data, **meta)
            self.viewer.add_image(flatfield)
            if self._settings["get_darkfield"].value:
                self.viewer.add_image(darkfield)

        @thread_worker(start_thread=False, connect={"returned": update_layer})
        def call_basic(data):
            basic = BaSiC(**self.settings)
            corrected = basic.fit_transform(data, timelapse=self._extrasettings["get_timelapse"].value)
            flatfield = basic.flatfield
            darkfield = basic.darkfield
            self.run_btn.setDisabled(False)  # reenable run button
            return corrected, flatfield, darkfield, meta

        worker = call_basic(data)
        self.cancel_btn.clicked.connect(partial(self._cancel, worker=worker))
        worker.finished.connect(self.cancel_btn.clicked.disconnect)
        worker.errored.connect(lambda: self.run_btn.setDisabled(False))
        worker.start()
        logger.info("BaSiC worker started")
        return worker

    def _cancel(self, worker):
        logger.info("Cancel requested")
        worker.quit()
        # enable run button
        worker.finished.connect(lambda: self.run_btn.setDisabled(False))

    def showEvent(self, event: QEvent) -> None:  # noqa: D102
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event: Optional[QEvent] = None) -> None:
        """Repopulate image layer dropdown list."""  # noqa DAR101
        self.image_select.reset_choices(event)
        self.weight_select.reset_choices(event)
        # If no layers are present, disable the 'run' button
        if len(self.image_select) <= 1:
            self.run_btn.setEnabled(False)
            self.autotune_btn.setEnabled(False)
        else:
            self.run_btn.setEnabled(True)
            self.autotune_btn.setEnabled(True)


class QScientificDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with scientific notation."""

    def __init__(self, *args, **kwargs):
        """Initialize a QDoubleSpinBox for scientific notation input."""
        super().__init__(*args, **kwargs)
        self.validator = QDoubleValidator()
        self.validator.setNotation(QDoubleValidator.ScientificNotation)
        self.setDecimals(10)
        self.setMinimum(-np.inf)
        self.setMaximum(np.inf)

    def validate(self, text, pos):  # noqa: D102
        return self.validator.validate(text, pos)

    def fixup(self, text):  # noqa: D102
        return self.validator.fixup(text)

    def textFromValue(self, value):  # noqa: D102
        return f"{value:.2E}"


class ScientificDoubleSpinBox:
    """Widget for inputing scientific notation."""

    def __init__(self, *args, **kwargs):
        """Initialize a scientific spinbox widget."""
        self.native = QScientificDoubleSpinBox(*args, **kwargs)

    @property
    def value(self):
        """Return the current value of the widget."""
        return self.native.value()
