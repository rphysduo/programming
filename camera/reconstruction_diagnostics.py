"""Explain stereo geometry flags without changing acceptance or coordinates."""

import math
from collections.abc import Mapping


def describe_point_flags(point: Mapping, *, reprojection_limit_px: float,
                         epipolar_limit_px: float,
                         rejected_parallax_deg: float) -> list[str]:
    """Describe existing flags and measured values, distinguishing camera/board Z.

    Limits must be the positive, finite values used by the triangulator. This
    function only explains its result; it does not decide whether a point passes.
    """
    limits = (reprojection_limit_px, epipolar_limit_px, rejected_parallax_deg)
    if not all(math.isfinite(value) and value > 0 for value in limits):
        raise ValueError("Geometry limits must be positive and finite")

    def measured(key):
        value = point.get(key)
        return f"{value:.3f}" if value is not None and math.isfinite(value) else "unavailable"

    messages = {
        "nonfinite": "Triangulation is non-finite or undefined.",
        "behind_cam0": (
            f"Behind cam0: camera depth Z0 = {measured('depth_z_cam0_mm')} mm; "
            "must be > 0. This is camera Z, not board Z."),
        "behind_cam1": (
            f"Behind cam1: camera depth Z1 = {measured('depth_z_cam1_mm')} mm; "
            "must be > 0. This is camera Z, not board Z."),
        "high_reproj_cam0": (
            f"Cam0 reprojection error {measured('reproj_error_cam0_px')} px "
            f"> {reprojection_limit_px:g} px limit."),
        "high_reproj_cam1": (
            f"Cam1 reprojection error {measured('reproj_error_cam1_px')} px "
            f"> {reprojection_limit_px:g} px limit."),
        "high_epipolar": (
            f"Epipolar error {measured('epipolar_sampson_px')} px "
            f"> {epipolar_limit_px:g} px limit."),
        "low_parallax": (
            f"Warning: small ray angle {measured('parallax_deg')} deg; "
            "depth is sensitive to pixel errors. This warning alone does not reject a point."),
        "rejected_parallel_rays": (
            f"Ray angle {measured('parallax_deg')} deg "
            f"< {rejected_parallax_deg:g} deg; depth is poorly constrained."),
        "geometry_error": f"Geometry computation failed: {point.get('geometry_error', 'unknown error')}",
    }
    reasons = [messages.get(flag, f"Geometry flag: {flag}")
               for flag in dict.fromkeys(point.get("flags", []))]
    if point.get("valid") is False and not reasons:
        reasons.append("Invalid geometry; no reason was returned.")
    return reasons
