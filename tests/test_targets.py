import textwrap
import warnings

import numpy as np
import pytest

from mjecv.calibration import (
    AprilGridTarget,
    CalibrationTarget,
    CalibrationTargetType,
    CheckerboardTarget,
    CircleGridTarget,
)

aprilgrid_yaml = textwrap.dedent(
    """\
    target_type: 'aprilgrid'
    tagCols: 6
    tagRows: 7
    tagSize: 0.088
    tagSpacing: 0.3
"""
)


def _check_aprilgrid(target):
    assert target.type_ == CalibrationTargetType.AprilGrid
    assert target.cols == 6
    assert target.rows == 7
    assert np.isclose(target.size, 0.088)
    assert np.isclose(target.spacing, 0.3)


def test_target_aprilgrid():
    target = CalibrationTarget.from_kalibr_yaml(aprilgrid_yaml)
    _check_aprilgrid(target)


@pytest.fixture
def aprilgrid_yaml_path(tmp_path):
    path = tmp_path / "aprilgrid.yaml"
    path.write_text(aprilgrid_yaml)
    return path


def test_target_aprilgrid_filename(aprilgrid_yaml_path):
    target = CalibrationTarget.from_kalibr_yaml(str(aprilgrid_yaml_path))
    _check_aprilgrid(target)


def test_target_aprilgrid_file(aprilgrid_yaml_path):
    target = CalibrationTarget.from_kalibr_yaml(aprilgrid_yaml_path.open())
    _check_aprilgrid(target)


def test_target_checkerboard():
    target_yaml = textwrap.dedent(
        """\
        target_type: 'checkerboard'
        targetCols: 6
        targetRows: 7
        rowSpacingMeters: 0.06
        colSpacingMeters: 0.06
    """
    )
    target = CalibrationTarget.from_kalibr_yaml(target_yaml)
    assert target.type_ == CalibrationTargetType.Checkerboard
    assert target.cols == 6
    assert target.rows == 7
    assert np.isclose(target.row_spacing, 0.06)
    assert np.isclose(target.col_spacing, 0.06)


def test_warn_rectangular_squares():
    with warnings.catch_warnings(record=True) as caught_warnings:
        CheckerboardTarget((8, 5), 0.1, 0.2)
        assert len(caught_warnings) == 1


def test_target_circlegrid():
    target_yaml = textwrap.dedent(
        """\
        target_type: 'circlegrid'
        targetCols: 6
        targetRows: 7
        spacingMeters: 0.02
        asymmetricGrid: False
    """
    )
    target = CalibrationTarget.from_kalibr_yaml(target_yaml)
    assert target.type_ == CalibrationTargetType.CircleGrid
    assert target.cols == 6
    assert target.rows == 7
    assert np.isclose(target.spacing, 0.02)
    assert target.asymmetric_grid is False


def test_target_create_aprilgrid():
    target = AprilGridTarget((7, 6), 0.088, 0.3)
    assert target.cols == 7
    assert target.rows == 6
    assert np.isclose(target.size, 0.088)
    assert np.isclose(target.spacing, 0.3)


def test_target_create_checkerboard_square():
    target = CheckerboardTarget((7, 6), 0.06)
    assert target.cols == 7
    assert target.rows == 6
    assert np.isclose(target.row_spacing, 0.06)
    assert np.isclose(target.col_spacing, 0.06)


def test_target_create_checkerboard_rect():
    target = CheckerboardTarget((7, 6), 0.06, 0.07)
    assert target.cols == 7
    assert target.rows == 6
    assert np.isclose(target.row_spacing, 0.06)
    assert np.isclose(target.col_spacing, 0.07)


def test_target_create_circlegrid():
    target = CircleGridTarget((7, 6), 0.02, False)
    assert target.cols == 7
    assert target.rows == 6
    assert np.isclose(target.spacing, 0.02)
    assert target.asymmetric_grid is False
