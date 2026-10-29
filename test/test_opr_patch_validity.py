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


class _ConstantModel:
    normalization = np.stack([np.zeros(5), np.ones(5)])

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, inputs, batch_size=None):
        n = inputs[0].shape[0]
        size = inputs[0].shape[1]
        return (np.zeros((n, size, size, 1)), np.full((n, size, size, 1), self.value))


def test_rejected_tile_is_nan_where_uncovered_and_never_overwrites() -> None:
    shape = (512, 512)
    invalidity_mask = np.zeros(shape, dtype=np.uint8)
    invalidity_mask[:, :] = 1
    invalidity_mask[:, :200] = 0
    invalidity_mask[:, 400:420] = 0
    inference = OPRInference(
        vv_nice_display=np.ones(shape),
        vh_nice_display=np.ones(shape),
        vh_nesz_in=np.zeros(shape),
        ecmwf_wind_speed=np.zeros(shape),
        incidence=np.zeros(shape),
        invalidity_mask=invalidity_mask,
        models=(),
    )
    result = inference.run_model(_ConstantModel(5.0))
    valid = [
        inference.padded_mask[c].mean() > 0.5 for c in inference.coordinates
    ]
    assert not all(valid) and any(valid)
    pad = inference.pad
    covered = np.zeros(inference.padded_shape[:2], dtype=bool)
    for c, v in zip(inference.coordinates, valid):
        if v:
            covered[c] = True
    covered = covered[pad:pad + shape[0], pad:pad + shape[1]]
    expected_valid = covered & (inference.mask > 0.5)
    assert np.all(result[expected_valid] == 5.0)
    assert np.all(np.isnan(result[~expected_valid]))
    assert (~covered & (inference.mask > 0.5)).any()
