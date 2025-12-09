from boolrepr.data import BooleanFunctionDataset


def test_data_shape():
    n_samp, seq, d_in = 1, 4, 2
    dataset = BooleanFunctionDataset(
        function_class="conjunction",
        num_samples=n_samp,
        seq_length=seq,
        input_dim=d_in,
        noise_prob=0,
        random_seed=1,
    )

    data = dataset.data
    x = data[0]["x"]
    y = data[0]["y"]

    assert len(data) == n_samp
    assert x.shape == (seq, d_in)
    assert y.shape == (seq,)
