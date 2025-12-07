from boolrepr.data import BooleanFunctionDataset


def test_data_shape():
    n_samp, seq, d_in = 1, 2, 2
    dataset = BooleanFunctionDataset(
        function_class="conjunction",
        num_samples=n_samp,
        seq_length=seq,
        input_dim=d_in,
        noise_prob=0,
        seed=1,
    )

    assert dataset.data.shape, (n_samp, 2 * seq, d_in + 1)
