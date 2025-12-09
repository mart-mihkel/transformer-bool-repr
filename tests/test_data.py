from boolrepr.data import BooleanFunctionDataset


def test_data_shape():
    d_in = 2
    seq = 2**d_in

    dataset = BooleanFunctionDataset(
        function_class="conjunction",
        input_dim=d_in,
        random_seed=1,
    )

    data = dataset.data
    x = data["x"]
    y = data["y"]

    assert len(data) == 2
    assert x.shape == (seq, d_in)
    assert y.shape == (seq,)
