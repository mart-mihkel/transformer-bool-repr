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
    x = data[0]["x"]

    assert len(data) == seq
    assert len(x) == d_in
