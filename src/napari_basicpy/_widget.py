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
)

if TYPE_CHECKING:
    import napari  # pragma: no cover

from magicgui.widgets import ComboBox
from napari.layers import Image

SHOW_LOGO = True  # Show or hide the BaSiC logo in the widget

logger = logging.getLogger(__name__)


def bool_layers(wdg: ComboBox) -> list[Image]:
    return [layer for layer in viewer.layers if isinstance(layer, Image)]


class BasicWidget(QWidget):
    """Example widget class."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Init example widget."""  # noqa DAR101
        super().__init__()

        self.viewer = viewer
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

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

        def build_layer_select_container():
            layer_select_container = QWidget()
            layer_select_layout = QFormLayout()
            layer_select_container.setLayout(layer_select_layout)
            layer_select_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            # Layer select should be accessible by BaSiC to access current layer
            self.layer_select = ComboBox(choices=bool_layers)
            layer_select_layout.addRow(self.layer_select.native)
            return layer_select_container

        def build_toggle_advanced_settings_cb():
            toggle_advanced_settings_cb = QCheckBox("Show Advanced Settings")
            return toggle_advanced_settings_cb

        def build_doc_reference_label():
            doc_reference_label = QLabel()
            doc_reference_label.setOpenExternalLinks(True)
            doc_reference_label.setText(
                '<a href="https://basicpy.readthedocs.io/en/latest/api.html#basicpy.basicpy.BaSiC">'
                "See docs for settings details</a>"
            )
            return doc_reference_label

        def build_settings_containers():
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
                widget.native.setMinimumWidth(150)
                return widget

            # All settings here will be used to initialize BaSiC
            self._settings = {k: build_widget(k) for k in BaSiC().settings.keys() if k not in skip}
            self._extrasettings = dict()
            self._extrasettings["get_timelapse"] = create_widget(
                value=False,
                options={"tooltip": "Output timelapse correction with corrected image"},
            )

            # build simple settings container
            simple_settings_gb = QGroupBox("Settings")  # make groupbox
            simple_settings_gb.setLayout(QVBoxLayout())
            simple_settings_form = QWidget()  # make form
            simple_settings_form.setLayout(QFormLayout())
            simple_settings_form.layout().setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            simple_settings_gb.layout().addWidget(simple_settings_form)  # add form to groupbox

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

            # add extra settings to simple settings container
            for k, v in self._extrasettings.items():
                simple_settings_form.layout().addRow(k, v.native)

            # The scroll view is created after filling the list to make it the correct size
            advanced_settings_scroll = QScrollArea()  # make scroll view
            advanced_settings_scroll.setWidget(advanced_settings_form)  # apply scroll view to form
            advanced_settings_gb.layout().addWidget(advanced_settings_scroll)  # add view to groupbox

            return simple_settings_gb, advanced_settings_gb

        # Build widget components
        header_container = build_header_container()
        layer_select_container = build_layer_select_container()
        doc_reference_lbl = build_doc_reference_label()
        (
            simple_settings_container,
            advanced_settings_container,
        ) = build_settings_containers()
        toggle_advanced_settings_cb = build_toggle_advanced_settings_cb()
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")

        # Add containers/widgets to layout
        main_layout.addWidget(header_container)
        main_layout.addWidget(layer_select_container)
        main_layout.addWidget(doc_reference_lbl)
        main_layout.addWidget(simple_settings_container)
        main_layout.addWidget(toggle_advanced_settings_cb)
        main_layout.addWidget(advanced_settings_container)
        main_layout.addWidget(self.run_btn)
        main_layout.addWidget(self.cancel_btn)

        # Show/hide widget components
        advanced_settings_container.setVisible(False)

        # Connect actions
        def toggle_advanced_settings():
            """Toggle the advanced settings container."""
            if toggle_advanced_settings_cb.isChecked():
                advanced_settings_container.setHidden(False)
            else:
                advanced_settings_container.setHidden(True)

        self.run_btn.clicked.connect(self._run)
        toggle_advanced_settings_cb.stateChanged.connect(toggle_advanced_settings)

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
        self.layer_select.reset_choices(event)
        # If no layers are present, disable the 'run' button
        if len(self.layer_select) < 1:
            self.run_btn.setEnabled(False)
        else:
            self.run_btn.setEnabled(True)


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
