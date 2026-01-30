"""
BigWarp-like interface for napari.

Provides point-based image registration using thin-plate spline transforms.
"""
import json
import magicgui
import napari
import numpy as np
from napari.layers import Image
from napari.utils.events import Event
from qtpy.QtWidgets import (
    QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
    QLabel, QSlider, QDoubleSpinBox, QGroupBox, QFormLayout, QComboBox
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSpinBox, QApplication
import pyqtgraph as pg


class BigWarpQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.model = BigWarpModel(self.viewer)

        self.fixed_selection_widget = magicgui.magicgui(
            self._select_fixed_layer,
            layer={"label": "Fixed", "choices": self.get_input_layers},
            auto_call=True,
        )

        self.moving_selection_widget = magicgui.magicgui(
            self._select_moving_layer,
            layer={"label": "Moving", "choices": self.get_input_layers},
            auto_call=True,
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.moving_selection_widget.native)
        self.layout().addWidget(self.fixed_selection_widget.native)

        # Registration method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Registration:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["TPS (Landmarks)", "Elastix (Auto)", "Hybrid (Landmarks + BSpline)"])
        self.method_combo.setToolTip(
            "TPS: Thin-plate spline from landmarks only\n"
            "Elastix: Fully automatic (no landmarks needed)\n"
            "Hybrid: Use landmarks for affine, then BSpline refinement"
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self.method_combo)
        method_widget = QWidget()
        method_widget.setLayout(method_layout)
        self.layout().addWidget(method_widget)

        # Landmark transform type selector (for TPS mode)
        landmark_layout = QHBoxLayout()
        landmark_layout.addWidget(QLabel("Landmark Transform:"))
        self.landmark_transform_combo = QComboBox()
        self.landmark_transform_combo.addItems(["TPS (Non-linear)", "Similarity (Scale+Rot)", "Affine", "Rigid"])
        self.landmark_transform_combo.setToolTip(
            "TPS: Non-linear thin-plate spline (good for tissue deformation)\n"
            "Similarity: Scale + Rotation + Translation (good for 10x↔100x)\n"
            "Affine: Similarity + Shear\n"
            "Rigid: Rotation + Translation only"
        )
        landmark_layout.addWidget(self.landmark_transform_combo)
        landmark_widget = QWidget()
        landmark_widget.setLayout(landmark_layout)
        self.layout().addWidget(landmark_widget)


        # Elastix settings group (hidden by default)
        self.elastix_group = QGroupBox("Elastix Settings")
        elastix_layout = QFormLayout()

        self.transform_combo = QComboBox()
        self.transform_combo.addItems(["Rigid", "Affine (Ri→Af)", "Composite (Ri→Af→BS)"])
        self.transform_combo.setCurrentIndex(2)  # Composite default
        self.transform_combo.setToolTip(
            "Rigid: rotation + translation only\n"
            "Affine: Rigid → Affine (adds scaling/shear)\n"
            "Composite: Rigid → Affine → BSpline (full nonlinear)"
        )
        elastix_layout.addRow("Transform:", self.transform_combo)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(50, 2000)
        self.iterations_spin.setValue(200)
        self.iterations_spin.setToolTip(
            "Max iterations per resolution level.\n"
            "Higher = more refined but slower.\n"
            "Start with 200, increase if not converging."
        )
        elastix_layout.addRow("Iterations:", self.iterations_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.1, 10.0)
        self.learning_rate_spin.setSingleStep(0.1)
        self.learning_rate_spin.setValue(1.0)
        self.learning_rate_spin.setToolTip(
            "Gradient descent step size.\n"
            "Higher = faster but may overshoot.\n"
            "Lower = more stable but slower.\n"
            "Default 1.0 works well for most cases."
        )
        elastix_layout.addRow("Learning Rate:", self.learning_rate_spin)

        self.sampling_spin = QDoubleSpinBox()
        self.sampling_spin.setRange(0.001, 1.0)
        self.sampling_spin.setDecimals(3)
        self.sampling_spin.setSingleStep(0.01)
        self.sampling_spin.setValue(0.01)
        self.sampling_spin.setToolTip(
            "Fraction of pixels used to compute metric.\n"
            "0.01 = 1% of pixels (fast, good for large images).\n"
            "Higher = more accurate but slower.\n"
            "Increase if registration is noisy."
        )
        elastix_layout.addRow("Sampling %:", self.sampling_spin)

        self.bspline_grid_spin = QDoubleSpinBox()
        self.bspline_grid_spin.setRange(10.0, 200.0)
        self.bspline_grid_spin.setSingleStep(10.0)
        self.bspline_grid_spin.setValue(50.0)
        self.bspline_grid_spin.setToolTip(
            "Spacing between BSpline control points (pixels).\n"
            "Smaller = more local deformation (finer detail).\n"
            "Larger = smoother, more global deformation.\n"
            "50-100 works well for most tissue images."
        )
        elastix_layout.addRow("Grid Spacing:", self.bspline_grid_spin)

        self.elastix_group.setLayout(elastix_layout)
        self.elastix_group.hide()  # Hidden by default (TPS mode)
        self.layout().addWidget(self.elastix_group)

        # Live optimization plot (pyqtgraph)
        self.plot_widget = pg.PlotWidget(title="Optimization")
        self.plot_widget.setLabel('left', 'Metric Value')
        self.plot_widget.setLabel('bottom', 'Iteration')
        self.plot_widget.setMaximumHeight(150)
        self.plot_widget.setBackground('w')
        self.plot_curve = self.plot_widget.plot(pen='g')
        self.metric_history = []
        self.plot_widget.hide()  # Show only during elastix
        self.layout().addWidget(self.plot_widget)

        # Add compute transform button
        self.compute_btn = QPushButton("Compute Transform")
        self.compute_btn.clicked.connect(self._compute_transform)
        self.layout().addWidget(self.compute_btn)

        # Add export/import points buttons
        btn_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Points")
        self.export_btn.clicked.connect(self._export_points)
        btn_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton("Import Points")
        self.import_btn.clicked.connect(self._import_points)
        btn_layout.addWidget(self.import_btn)

        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        self.layout().addWidget(btn_widget)

        # Add separate import buttons for fixed and moving from NPZ
        import_npz_layout = QHBoxLayout()
        self.import_fixed_btn = QPushButton("Import Fixed (NPZ)")
        self.import_fixed_btn.clicked.connect(self._import_fixed_points)
        import_npz_layout.addWidget(self.import_fixed_btn)

        self.import_moving_btn = QPushButton("Import Moving (NPZ)")
        self.import_moving_btn.clicked.connect(self._import_moving_points)
        import_npz_layout.addWidget(self.import_moving_btn)

        import_npz_widget = QWidget()
        import_npz_widget.setLayout(import_npz_layout)
        self.layout().addWidget(import_npz_widget)

        # Add export transform button
        self.export_transform_btn = QPushButton("Export Transform")
        self.export_transform_btn.clicked.connect(self._export_transform)
        self.layout().addWidget(self.export_transform_btn)

        # Add export deformation field button
        self.export_deform_btn = QPushButton("Export Deformation Field")
        self.export_deform_btn.clicked.connect(self._export_deformation_field)
        self.layout().addWidget(self.export_deform_btn)

        # Add export transformed image button
        self.export_image_btn = QPushButton("Export Transformed Image")
        self.export_image_btn.clicked.connect(self._export_transformed_image)
        self.layout().addWidget(self.export_image_btn)

        # Add auto-detect landmarks section
        detect_group = QGroupBox("Auto-Detect Settings")
        detect_layout = QFormLayout()

        # RANSAC model dropdown
        self.ransac_model_combo = QComboBox()
        self.ransac_model_combo.addItems(["affine", "rigid", "homography"])
        detect_layout.addRow("RANSAC Model:", self.ransac_model_combo)

        # Ratio threshold (for descriptor matching)
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.5, 0.95)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setValue(0.75)
        self.ratio_spin.setToolTip("Lowe's ratio test threshold. Higher = more matches, lower = stricter")
        detect_layout.addRow("Match Ratio:", self.ratio_spin)

        # RANSAC threshold
        self.ransac_thresh_spin = QDoubleSpinBox()
        self.ransac_thresh_spin.setRange(1.0, 100.0)
        self.ransac_thresh_spin.setSingleStep(5.0)
        self.ransac_thresh_spin.setValue(15.0)
        self.ransac_thresh_spin.setToolTip("RANSAC reprojection threshold in pixels. Higher = more lenient")
        detect_layout.addRow("RANSAC Thresh (px):", self.ransac_thresh_spin)

        detect_group.setLayout(detect_layout)
        self.layout().addWidget(detect_group)

        # Add auto-detect landmarks button
        self.auto_detect_btn = QPushButton("Auto-Detect Landmarks")
        self.auto_detect_btn.clicked.connect(self._auto_detect_landmarks)
        self.layout().addWidget(self.auto_detect_btn)

        # Add reset buttons
        reset_layout = QHBoxLayout()
        self.reset_transform_btn = QPushButton("Reset Transform")
        self.reset_transform_btn.clicked.connect(self._reset_transform)
        reset_layout.addWidget(self.reset_transform_btn)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self._clear_points)
        reset_layout.addWidget(self.clear_points_btn)

        reset_widget = QWidget()
        reset_widget.setLayout(reset_layout)
        self.layout().addWidget(reset_widget)

        self.viewer.layers.events.inserted.connect(self.moving_selection_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.moving_selection_widget.reset_choices)
        self.viewer.layers.events.inserted.connect(self.fixed_selection_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.fixed_selection_widget.reset_choices)

        self.fixed_selection_widget.reset_choices()
        self.moving_selection_widget.reset_choices()

        # Register 'T' keybinding for transform
        @self.viewer.bind_key('t')
        def _trigger_transform(viewer):
            """Press T to compute transform."""
            self._compute_transform()

    def _select_moving_layer(self, layer: str):
        self.model.moving_layer_name = layer

    def _select_fixed_layer(self, layer: str):
        self.model.fixed_layer_name = layer

    def get_input_layers(self, _):
        return [""] + [x.name for x in self.viewer.layers if isinstance(x, Image) and not x.name.startswith("[BW]")]

    def _on_method_changed(self, index):
        """Handle registration method change."""
        # Show elastix settings for Elastix (1) and Hybrid (2) modes
        uses_elastix = index >= 1
        self.elastix_group.setVisible(uses_elastix)

    def _update_plot(self, metric, iteration, new_level=False):
        """Callback from elastix registration to update live plot."""
        if new_level:
            # Mark resolution change with vertical line
            self.plot_widget.addLine(x=len(self.metric_history), pen='r')
        elif metric is not None:
            self.metric_history.append(metric)
            self.plot_curve.setData(self.metric_history)
            QApplication.processEvents()  # Update UI during registration

    def _compute_transform(self):
        """Manually trigger transform computation."""
        mode = self.method_combo.currentIndex()
        if mode == 0:
            # Landmark mode - use selected transform type
            landmark_type_idx = self.landmark_transform_combo.currentIndex()
            landmark_types = ['tps', 'similarity', 'affine', 'rigid']
            transform_type = landmark_types[landmark_type_idx]
            self.model.compute_transform(transform_type=transform_type)
        elif mode == 1:
            # Elastix mode - automatic registration
            self._compute_elastix_transform()
        else:
            # Hybrid mode - landmarks for affine, then BSpline refinement
            self._compute_hybrid_transform()

    def _compute_elastix_transform(self):
        """Run elastix registration on the selected images."""
        if not self.model.fixed_layer_name or not self.model.moving_layer_name:
            napari.utils.notifications.show_warning("Please select fixed and moving layers first.")
            return

        try:
            fixed_img = self.model.fixed_layer.data
            moving_img = self.model.moving_layer.data
        except KeyError:
            napari.utils.notifications.show_warning("Selected layers not found.")
            return

        # Get parameters from UI
        # Map display names to internal transform types
        transform_display = self.transform_combo.currentText()
        if "Composite" in transform_display:
            transform_type = "bspline"  # composite uses full Rigid→Affine→BSpline pipeline
        elif "Affine" in transform_display:
            transform_type = "affine"  # Rigid→Affine pipeline
        else:
            transform_type = "rigid"  # Rigid only
        
        params = {
            'iterations': self.iterations_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'sampling_percentage': self.sampling_spin.value(),
            'bspline_grid_spacing': self.bspline_grid_spin.value(),
        }

        # Clear and show plot
        self.metric_history = []
        self.plot_curve.clear()
        self.plot_widget.clear()
        self.plot_curve = self.plot_widget.plot(pen='g')
        self.plot_widget.show()

        napari.utils.notifications.show_info(f"Running {transform_type} registration... Watch the optimization curve.")

        try:
            from napari_bigwarp.elastix_registration import register_images

            # Run registration with callbacks
            def progress_cb(metric, iteration):
                self._update_plot(metric, iteration)

            def level_cb(level):
                self._update_plot(None, None, new_level=True)

            result, transform = register_images(
                fixed_img, moving_img,
                transform_type=transform_type,
                params=params,
                progress_callback=progress_cb,
                level_callback=level_cb,
            )

            # Update the moving result layer
            if self.model.moving_result_layer is not None:
                self.model.moving_result_layer.data = result
            else:
                napari.utils.notifications.show_warning("Moving result layer not initialized.")

            # Store the transform for potential export
            self._elastix_transform = transform

            napari.utils.notifications.show_info("Elastix registration complete!")

        except ImportError as e:
            napari.utils.notifications.show_error(f"Missing dependency: {e}. Install with: pip install SimpleITK")
        except Exception as e:
            napari.utils.notifications.show_error(f"Registration failed: {e}")

    def _compute_hybrid_transform(self):
        """Run hybrid registration: landmarks for affine + BSpline refinement."""
        # Check layers are selected
        if not self.model.fixed_layer_name or not self.model.moving_layer_name:
            napari.utils.notifications.show_warning("Please select fixed and moving layers first.")
            return

        # Check we have landmark points
        if self.model.fixed_points_layer is None or self.model.moving_points_layer is None:
            napari.utils.notifications.show_warning("Point layers not initialized. Select layers first.")
            return

        fixed_pts = self.model.fixed_points_layer.data
        moving_pts = self.model.moving_points_layer.data

        if len(fixed_pts) < 3 or len(moving_pts) < 3:
            napari.utils.notifications.show_warning(
                f"Need at least 3 landmarks for hybrid mode. Fixed: {len(fixed_pts)}, Moving: {len(moving_pts)}"
            )
            return

        if len(fixed_pts) != len(moving_pts):
            napari.utils.notifications.show_warning(
                f"Point count mismatch. Fixed: {len(fixed_pts)}, Moving: {len(moving_pts)}"
            )
            return

        try:
            fixed_img = self.model.fixed_layer.data
            moving_img = self.model.moving_layer.data
        except KeyError:
            napari.utils.notifications.show_warning("Selected layers not found.")
            return

        # Get BSpline parameters from UI
        params = {
            'iterations': self.iterations_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'sampling_percentage': self.sampling_spin.value(),
            'bspline_grid_spacing': self.bspline_grid_spin.value(),
        }

        # Clear and show plot
        self.metric_history = []
        self.plot_curve.clear()
        self.plot_widget.clear()
        self.plot_curve = self.plot_widget.plot(pen='g')
        self.plot_widget.show()

        napari.utils.notifications.show_info(
            f"Hybrid mode: Using {len(fixed_pts)} landmarks for affine, then BSpline refinement..."
        )

        try:
            from napari_bigwarp.elastix_registration import register_images_with_landmarks

            def progress_cb(metric, iteration):
                self._update_plot(metric, iteration)

            def level_cb(level):
                self._update_plot(None, None, new_level=True)

            result, transform = register_images_with_landmarks(
                fixed_img, moving_img,
                fixed_pts, moving_pts,
                params=params,
                progress_callback=progress_cb,
                level_callback=level_cb,
            )

            # Update the moving result layer
            if self.model.moving_result_layer is not None:
                self.model.moving_result_layer.data = result
            else:
                napari.utils.notifications.show_warning("Moving result layer not initialized.")

            # Store the transform for potential export
            self._elastix_transform = transform

            napari.utils.notifications.show_info("Hybrid registration complete!")

        except ImportError as e:
            napari.utils.notifications.show_error(f"Missing dependency: {e}. Install with: pip install SimpleITK")
        except Exception as e:
            napari.utils.notifications.show_error(f"Hybrid registration failed: {e}")

    def _export_points(self):
        """Export landmark points to a JSON file."""
        if self.model.fixed_points_layer is None or self.model.moving_points_layer is None:
            napari.utils.notifications.show_warning("No points to export. Place some landmarks first.")
            return

        fixed_pts = self.model.fixed_points_layer.data
        moving_pts = self.model.moving_points_layer.data

        if len(fixed_pts) == 0 and len(moving_pts) == 0:
            napari.utils.notifications.show_warning("No points to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Landmarks", "", "JSON files (*.json);;CSV files (*.csv)"
        )
        if not filepath:
            return

        # Get scale info from layers if available
        fixed_scale = list(self.model.fixed_layer.scale) if self.model.fixed_layer is not None else [1.0, 1.0]
        moving_scale = list(self.model.moving_layer.scale) if self.model.moving_layer is not None else [1.0, 1.0]

        # Convert pixel coords to world coords (physical units)
        fixed_pts_world = fixed_pts * np.array(fixed_scale) if len(fixed_pts) > 0 else np.array([])
        moving_pts_world = moving_pts * np.array(moving_scale) if len(moving_pts) > 0 else np.array([])

        data = {
            "fixed_points_pixels": fixed_pts.tolist(),
            "moving_points_pixels": moving_pts.tolist(),
            "fixed_points_world": fixed_pts_world.tolist() if len(fixed_pts_world) > 0 else [],
            "moving_points_world": moving_pts_world.tolist() if len(moving_pts_world) > 0 else [],
            "fixed_scale": fixed_scale,
            "moving_scale": moving_scale,
            "fixed_layer_name": self.model.fixed_layer_name,
            "moving_layer_name": self.model.moving_layer_name,
        }

        if filepath.endswith(".csv"):
            # Export as CSV with columns for both pixel and world coordinates
            import csv
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "fixed_y_px", "fixed_x_px", "fixed_y_world", "fixed_x_world",
                    "moving_y_px", "moving_x_px", "moving_y_world", "moving_x_world"
                ])
                max_len = max(len(fixed_pts), len(moving_pts))
                for i in range(max_len):
                    row = []
                    if i < len(fixed_pts):
                        row.extend(fixed_pts[i].tolist())
                        row.extend(fixed_pts_world[i].tolist())
                    else:
                        row.extend(["", "", "", ""])
                    if i < len(moving_pts):
                        row.extend(moving_pts[i].tolist())
                        row.extend(moving_pts_world[i].tolist())
                    else:
                        row.extend(["", "", "", ""])
                    writer.writerow(row)
        else:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        napari.utils.notifications.show_info(f"Exported {len(fixed_pts)} fixed and {len(moving_pts)} moving points.")

    def _import_points(self):
        """Import landmark points from a JSON or NPZ file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Landmarks", "", 
            "All supported (*.json *.npz);;JSON files (*.json);;NumPy NPZ files (*.npz);;All files (*)"
        )
        if not filepath:
            return

        try:
            if filepath.endswith('.npz'):
                # Load from NPZ file
                data = np.load(filepath)
                
                # Try different key naming conventions
                fixed_pts = None
                moving_pts = None
                
                # Check for various key names
                for key in ['fixed', 'fixed_points', 'fixed_points_pixels', 'pts_fixed']:
                    if key in data:
                        fixed_pts = data[key]
                        break
                
                for key in ['moving', 'moving_points', 'moving_points_pixels', 'pts_moving']:
                    if key in data:
                        moving_pts = data[key]
                        break
                
                if fixed_pts is not None and self.model.fixed_points_layer is not None:
                    self.model.fixed_points_layer.data = np.array(fixed_pts)
                if moving_pts is not None and self.model.moving_points_layer is not None:
                    self.model.moving_points_layer.data = np.array(moving_pts)
                
                n_fixed = len(fixed_pts) if fixed_pts is not None else 0
                n_moving = len(moving_pts) if moving_pts is not None else 0
                napari.utils.notifications.show_info(
                    f"Imported {n_fixed} fixed and {n_moving} moving points from NPZ"
                )
                
            else:
                # Load from JSON file
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Handle both old format (fixed_points) and new format (fixed_points_pixels)
                fixed_key = "fixed_points_pixels" if "fixed_points_pixels" in data else "fixed_points"
                moving_key = "moving_points_pixels" if "moving_points_pixels" in data else "moving_points"

                if self.model.fixed_points_layer is not None and fixed_key in data:
                    self.model.fixed_points_layer.data = np.array(data[fixed_key])
                if self.model.moving_points_layer is not None and moving_key in data:
                    self.model.moving_points_layer.data = np.array(data[moving_key])

                napari.utils.notifications.show_info(f"Imported landmarks from {filepath}")
                
        except Exception as e:
            napari.utils.notifications.show_error(f"Failed to import landmarks: {e}")

    def _import_fixed_points(self):
        """Import fixed points from a single NPZ file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Fixed Points", "",
            "NumPy files (*.npz *.npy);;All files (*)"
        )
        if not filepath:
            return

        try:
            if self.model.fixed_points_layer is None:
                napari.utils.notifications.show_warning("Fixed points layer not initialized. Select layers first.")
                return

            if filepath.endswith('.npy'):
                pts = np.load(filepath)
            else:
                data = np.load(filepath)
                # Get the first array in the NPZ file
                keys = list(data.keys())
                if not keys:
                    napari.utils.notifications.show_error("NPZ file is empty")
                    return
                pts = data[keys[0]]

            # Swap from (X, Y) to (Y, X) for Napari (row, col)
            pts = np.array(pts)[:, ::-1]
            self.model.fixed_points_layer.data = pts
            napari.utils.notifications.show_info(f"Imported {len(pts)} fixed points from {filepath} (swapped X↔Y)")

        except Exception as e:
            napari.utils.notifications.show_error(f"Failed to import fixed points: {e}")

    def _import_moving_points(self):
        """Import moving points from a single NPZ file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Moving Points", "",
            "NumPy files (*.npz *.npy);;All files (*)"
        )
        if not filepath:
            return

        try:
            if self.model.moving_points_layer is None:
                napari.utils.notifications.show_warning("Moving points layer not initialized. Select layers first.")
                return

            if filepath.endswith('.npy'):
                pts = np.load(filepath)
            else:
                data = np.load(filepath)
                # Get the first array in the NPZ file
                keys = list(data.keys())
                if not keys:
                    napari.utils.notifications.show_error("NPZ file is empty")
                    return
                pts = data[keys[0]]

            # Swap from (X, Y) to (Y, X) for Napari (row, col)
            pts = np.array(pts)[:, ::-1]
            self.model.moving_points_layer.data = pts
            napari.utils.notifications.show_info(f"Imported {len(pts)} moving points from {filepath} (swapped X↔Y)")

        except Exception as e:
            napari.utils.notifications.show_error(f"Failed to import moving points: {e}")

    def _export_transform(self):
        """Export the transformation to a file."""
        # Check if we have an elastix transform
        if hasattr(self, '_elastix_transform') and self._elastix_transform is not None:
            # Export elastix transform as SimpleITK .tfm file
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Elastix Transform", "", "Transform files (*.tfm);;All files (*)"
            )
            if not filepath:
                return
            if not filepath.endswith('.tfm'):
                filepath += '.tfm'
            try:
                from napari_bigwarp.elastix_registration import save_transform
                save_transform(self._elastix_transform, filepath)
                napari.utils.notifications.show_info(f"Exported elastix transform to {filepath}")
            except Exception as e:
                napari.utils.notifications.show_error(f"Failed to export transform: {e}")
            return

        # TPS mode - export landmarks as JSON
        if self.model.fixed_points_layer is None or self.model.moving_points_layer is None:
            napari.utils.notifications.show_warning("No transform to export. Compute transform first.")
            return

        fixed_pts = self.model.fixed_points_layer.data
        moving_pts = self.model.moving_points_layer.data

        if len(fixed_pts) < 3 or len(moving_pts) < 3:
            napari.utils.notifications.show_warning("Need at least 3 point pairs to export transform.")
            return

        if len(fixed_pts) != len(moving_pts):
            napari.utils.notifications.show_warning(
                f"Point count mismatch. Fixed: {len(fixed_pts)}, Moving: {len(moving_pts)}"
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Transform", "", "JSON files (*.json)"
        )
        if not filepath:
            return

        # Get image metadata
        fixed_shape = list(self.model.fixed_layer.data.shape) if self.model.fixed_layer is not None else None
        moving_shape = list(self.model.moving_layer.data.shape) if self.model.moving_layer is not None else None
        fixed_scale = list(self.model.fixed_layer.scale) if self.model.fixed_layer is not None else [1.0, 1.0]
        moving_scale = list(self.model.moving_layer.scale) if self.model.moving_layer is not None else [1.0, 1.0]

        # Build the transform file
        # The TPS transform is fully defined by the control points
        transform_data = {
            "transform_type": "ThinPlateSpline",
            "fixed_points": fixed_pts.tolist(),
            "moving_points": moving_pts.tolist(),
            "fixed_image_shape": fixed_shape,
            "moving_image_shape": moving_shape,
            "fixed_scale": fixed_scale,
            "moving_scale": moving_scale,
            "fixed_layer_name": self.model.fixed_layer_name,
            "moving_layer_name": self.model.moving_layer_name,
            "description": "TPS transform: maps moving image to fixed image coordinate space",
        }

        with open(filepath, "w") as f:
            json.dump(transform_data, f, indent=2)

        napari.utils.notifications.show_info(f"Exported TPS transform with {len(fixed_pts)} control points.")

    def _export_deformation_field(self):
        """Export the dense deformation field (displacement at every pixel)."""
        if self.model.fixed_points_layer is None or self.model.moving_points_layer is None:
            napari.utils.notifications.show_warning("No transform computed. Place points and compute transform first.")
            return

        fixed_pts = self.model.fixed_points_layer.data
        moving_pts = self.model.moving_points_layer.data

        if len(fixed_pts) < 3 or len(moving_pts) < 3:
            napari.utils.notifications.show_warning("Need at least 3 point pairs to compute deformation field.")
            return

        if len(fixed_pts) != len(moving_pts):
            napari.utils.notifications.show_warning(
                f"Point count mismatch. Fixed: {len(fixed_pts)}, Moving: {len(moving_pts)}"
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Deformation Field", "", 
            "TIFF files (*.tiff *.tif);;NumPy files (*.npy)"
        )
        if not filepath:
            return

        napari.utils.notifications.show_info("Computing deformation field... This may take a moment.")

        # Get image shapes
        fixed_shape = self.model.fixed_layer.data.shape[:2]
        moving_shape = self.model.moving_layer.data.shape[:2]

        # Compute deformation field
        from napari_bigwarp.bigwarp import compute_deformation_field, save_deformation_field
        displacement_y, displacement_x = compute_deformation_field(
            fixed_pts, moving_pts, fixed_shape, moving_shape
        )

        # Save
        save_deformation_field(displacement_y, displacement_x, filepath)

        napari.utils.notifications.show_info(
            f"Exported deformation field ({fixed_shape[0]}x{fixed_shape[1]}) to {filepath}"
        )

    def _export_transformed_image(self):
        """Export the transformed moving image to a file."""
        if self.model.moving_result_layer is None:
            napari.utils.notifications.show_warning("No transformed image to export. Compute transform first.")
            return

        result_data = self.model.moving_result_layer.data
        if result_data is None or result_data.size == 0:
            napari.utils.notifications.show_warning("Transformed image is empty.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Transformed Image", "",
            "TIFF files (*.tiff *.tif);;PNG files (*.png);;All files (*)"
        )
        if not filepath:
            return

        try:
            import tifffile
            
            # Ensure proper file extension
            if not any(filepath.endswith(ext) for ext in ['.tiff', '.tif', '.png']):
                filepath += '.tif'
            
            if filepath.endswith('.png'):
                from skimage.io import imsave
                # Normalize to 8-bit for PNG
                img = result_data.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                imsave(filepath, img.astype(np.uint8))
            else:
                # Save as TIFF (preserves dtype)
                tifffile.imwrite(filepath, result_data)
            
            napari.utils.notifications.show_info(
                f"Exported transformed image ({result_data.shape[0]}x{result_data.shape[1]}) to {filepath}"
            )
        except Exception as e:
            napari.utils.notifications.show_error(f"Failed to export image: {e}")

    def _auto_detect_landmarks(self):
        """Automatically detect and match landmarks using skeleton-based detection."""
        if not self.model.fixed_layer_name or not self.model.moving_layer_name:
            napari.utils.notifications.show_warning("Please select fixed and moving layers first.")
            return

        if self.model.fixed_points_layer is None or self.model.moving_points_layer is None:
            napari.utils.notifications.show_warning("Point layers not initialized. Select layers first.")
            return

        napari.utils.notifications.show_info("Detecting landmarks... This may take a moment.")

        try:
            from napari_bigwarp.auto_landmark import auto_detect_landmarks

            fixed_img = self.model.fixed_layer.data
            moving_img = self.model.moving_layer.data

            # Get parameters from UI
            ransac_model = self.ransac_model_combo.currentText()
            ratio_thresh = self.ratio_spin.value()
            ransac_thresh = self.ransac_thresh_spin.value()

            # Run auto-detection with UI parameters
            fixed_pts, moving_pts, info = auto_detect_landmarks(
                fixed_img, moving_img,
                detect_endpoints=True,
                detect_bifurcations=True,
                ransac_model=ransac_model,
                ransac_thresh=ransac_thresh,
                min_matches=4,
                ratio_thresh=ratio_thresh,
            )

            if 'error' in info:
                napari.utils.notifications.show_warning(f"Auto-detect failed: {info['error']}")
                return

            if len(fixed_pts) < 3:
                napari.utils.notifications.show_warning(
                    f"Not enough landmarks detected after RANSAC ({len(fixed_pts)} points). "
                    f"Before RANSAC: {info.get('matches_before_ransac', 0)}. "
                    f"Try increasing Match Ratio or RANSAC Thresh."
                )
                return

            # Populate point layers
            self.model.fixed_points_layer.data = fixed_pts
            self.model.moving_points_layer.data = moving_pts

            napari.utils.notifications.show_info(
                f"Detected {len(fixed_pts)} landmarks! "
                f"(Keypoints: F={info['keypoints_fixed']}, M={info['keypoints_moving']}, "
                f"Matches: {info['matches_before_ransac']}→{info['matches_after_ransac']} after RANSAC)"
            )

        except ImportError as e:
            napari.utils.notifications.show_error(f"Missing dependency: {e}")
        except Exception as e:
            napari.utils.notifications.show_error(f"Auto-detect error: {e}")

    def _reset_transform(self):
        """Reset the moving result layer to show the original (unwarped) moving image."""
        if self.model.moving_result_layer is None or self.model.moving_layer is None:
            napari.utils.notifications.show_warning("No transform to reset.")
            return

        # Reset moving result to original moving image data
        self.model.moving_result_layer.data = self.model.moving_layer.data.copy()
        napari.utils.notifications.show_info("Transform reset. Moving result shows original moving image.")

    def _clear_points(self):
        """Clear all landmark points from both layers."""
        if self.model.fixed_points_layer is not None:
            self.model.fixed_points_layer.data = np.zeros((0, 2), dtype=np.float32)
        if self.model.moving_points_layer is not None:
            self.model.moving_points_layer.data = np.zeros((0, 2), dtype=np.float32)
        napari.utils.notifications.show_info("All points cleared.")


class BigWarpModel:
    def __init__(self, viewer: napari.Viewer):
        self._viewer = viewer
        self._moving_layer_name = ""
        self._fixed_layer_name = ""
        self.fixed_result_layer = None
        self.moving_result_layer = None
        self.fixed_points_layer = None
        self.moving_points_layer = None

    def _update_layers(self):
        if not self.moving_layer_name or not self.fixed_layer_name:
            return

        self.fixed_layer.translate = [0, 0]
        self.moving_layer.translate = [0, self.fixed_layer.data.shape[1]]

        if self.fixed_result_layer is None:
            self.fixed_result_layer = self._viewer.add_image(
                self.fixed_layer.data,
                name="[BW] Fixed result",
                translate=[
                    0,
                    self.moving_layer.data.shape[1] + self.fixed_layer.data.shape[1],
                ],
                colormap="red",
            )
        else:
            self.fixed_result_layer.data = self.fixed_layer.data

        if self.moving_result_layer is None:
            self.moving_result_layer = self._viewer.add_image(
                self.moving_layer.data,
                name="[BW] Moving result",
                translate=self.fixed_result_layer.translate,
                blending="additive",
                colormap="green",
            )
        else:
            self.moving_result_layer.data = self.moving_layer.data

        if self.fixed_points_layer is None:
            self.fixed_points_layer = self._viewer.add_points(
                name="[BW] Fixed points",
                face_color="red",
                border_width=0.5,
                size=5,
                ndim=2,
                translate=self.fixed_layer.translate,
            )
        else:
            with self.fixed_points_layer.events.data.blocker():
                self.fixed_points_layer.data = np.zeros((0, 2), dtype=self.fixed_points_layer.data.dtype)

        if self.moving_points_layer is None:
            self.moving_points_layer = self._viewer.add_points(
                name="[BW] Moving points",
                face_color="green",
                border_width=0.5,
                size=5,
                ndim=2,
                translate=self.moving_layer.translate,
            )
        else:
            with self.moving_points_layer.events.data.blocker():
                self.moving_points_layer.data = np.zeros((0, 2), dtype=self.moving_points_layer.data.dtype)

        self.moving_points_layer.mode = "add"
        self.moving_points_layer.events.data.connect(self.on_add_point)

        visible_layers = [
            self.fixed_layer,
            self.moving_layer,
            self.fixed_result_layer,
            self.moving_result_layer,
            self.fixed_points_layer,
            self.moving_points_layer,
        ]
        for layer in self._viewer.layers:
            layer.visible = layer in visible_layers

        self._viewer.reset_view()

    def on_add_point(self, event: Event):
        # Only process ADDED events (not ADDING, REMOVING, etc.)
        # The action attribute may not exist in older napari versions
        action = getattr(event, 'action', None)
        if action is not None:
            # In newer napari, action is an enum; check for 'added' or ActionType.ADDED
            action_str = str(action).lower()
            if 'added' not in action_str:
                return
        
        # Check if there's data to process
        if event.source.data.shape[0] == 0:
            return
            
        last_point_world = event.source.data_to_world(event.source.data[-1])
        moving_value = self.moving_layer.get_value(last_point_world, world=True)
        fixed_value = self.fixed_layer.get_value(last_point_world, world=True)

        with event.source.events.data.blocker():
            event.source.selected_data = {event.source.data.shape[0] - 1}
            event.source.remove_selected()

        if moving_value is not None:
            add_to_layer = self.moving_points_layer
        elif fixed_value is not None:
            add_to_layer = self.fixed_points_layer
        else:
            return

        with add_to_layer.events.data.blocker():
            add_to_layer.add(add_to_layer.world_to_data(last_point_world))
        # Transform is NOT auto-triggered; user must press 'T' or click "Compute Transform"

    def compute_transform(self, transform_type: str = 'tps'):
        """Compute transform from current landmark points.
        
        Parameters
        ----------
        transform_type : str
            One of 'tps', 'similarity', 'affine', 'rigid'
        """
        if self.fixed_points_layer is None or self.moving_points_layer is None:
            napari.utils.notifications.show_warning("Please select fixed and moving layers first.")
            return

        fixed_pts = self.fixed_points_layer.data
        moving_pts = self.moving_points_layer.data

        min_points = 2 if transform_type == 'rigid' else 3
        if len(fixed_pts) < min_points or len(moving_pts) < min_points:
            napari.utils.notifications.show_warning(
                f"Need at least {min_points} points in each layer. Fixed: {len(fixed_pts)}, Moving: {len(moving_pts)}"
            )
            return

        if len(fixed_pts) != len(moving_pts):
            napari.utils.notifications.show_warning(
                f"Point count mismatch. Fixed: {len(fixed_pts)}, Moving: {len(moving_pts)}"
            )
            return

        if transform_type == 'tps':
            from napari_bigwarp.bigwarp import bigwarp
            result = bigwarp(
                fixed=self.fixed_layer.data,
                moving=self.moving_layer.data,
                fixed_points=fixed_pts,
                moving_points=moving_pts,
            )
            self.moving_result_layer.data = result
            self._linear_transform = None
            napari.utils.notifications.show_info(f"TPS transform computed using {len(fixed_pts)} landmarks.")
        else:
            # Linear transforms (similarity, affine, rigid)
            from napari_bigwarp.bigwarp import bigwarp_linear
            result, transform_info = bigwarp_linear(
                fixed=self.fixed_layer.data,
                moving=self.moving_layer.data,
                fixed_points=fixed_pts,
                moving_points=moving_pts,
                transform_type=transform_type,
            )
            self.moving_result_layer.data = result
            self._linear_transform = transform_info
            
            # Report transform parameters
            if transform_type == 'similarity':
                napari.utils.notifications.show_info(
                    f"Similarity transform: scale={transform_info['scale']:.4f}, "
                    f"rotation={np.degrees(transform_info['rotation']):.2f}°, "
                    f"error={transform_info['mean_error']:.2f}px"
                )
            elif transform_type == 'affine':
                napari.utils.notifications.show_info(
                    f"Affine transform: scale={transform_info['scale']}, "
                    f"error={transform_info['mean_error']:.2f}px"
                )
            else:
                napari.utils.notifications.show_info(
                    f"Rigid transform: rotation={np.degrees(transform_info['rotation']):.2f}°, "
                    f"error={transform_info['mean_error']:.2f}px"
                )

    @property
    def moving_layer(self) -> Image:
        return self._viewer.layers[self._moving_layer_name]

    @property
    def fixed_layer(self) -> Image:
        return self._viewer.layers[self._fixed_layer_name]

    @property
    def moving_layer_name(self) -> str:
        return self._moving_layer_name

    @moving_layer_name.setter
    def moving_layer_name(self, layer_name: str):
        self._moving_layer_name = layer_name
        self._update_layers()

    @property
    def fixed_layer_name(self) -> str:
        return self._fixed_layer_name

    @fixed_layer_name.setter
    def fixed_layer_name(self, layer_name: str):
        self._fixed_layer_name = layer_name
        self._update_layers()

