import numpy as np

from s1tools.opr_predictor.opr_inference import OPRInference


def test_patch_validity_uses_patch_coordinates() -> None:
    shape = (512, 512)
    invalidity_mask = np.ones(shape, dtype=np.uint8)
    invalidity_mask[:256, :256] = 0
    inference = OPRInference(
        vv_nice_display=np.ones(shape),
        vh_nice_display=np.ones(shape),
        vh_nesz_in=np.zeros(shape),
        ecmwf_wind_speed=np.zeros(shape),
        incidence=np.zeros(shape),
        invalidity_mask=invalidity_mask,
        models=(),
    )
    pad = inference.pad
    assert inference.padded_mask.shape == inference.padded_shape[:2]
    n_checked = 0
    for x_slice, y_slice in inference.coordinates:
        if x_slice.start < pad or y_slice.start < pad:
            continue
        if x_slice.stop > pad + shape[0] or y_slice.stop > pad + shape[1]:
            continue
        unpadded_window = inference.mask[
            x_slice.start - pad:x_slice.stop - pad,
            y_slice.start - pad:y_slice.stop - pad,
        ]
        padded_window = inference.padded_mask[x_slice, y_slice]
        assert np.array_equal(padded_window, unpadded_window)
        n_checked += 1
    assert n_checked > 0
