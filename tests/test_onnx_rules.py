from neurogolf.onnx_rules import check_file_size, find_external_data_files


def test_check_file_size() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "small.onnx"
        path.write_bytes(b"abcde")

        file_size, is_valid = check_file_size(path, max_file_size_bytes=10)
        assert file_size == 5
        assert is_valid is True

        _, is_valid_tight = check_file_size(path, max_file_size_bytes=3)
        assert is_valid_tight is False


def test_find_external_data_files() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "model.onnx"
        model_path.write_bytes(b"onnx")

        assert find_external_data_files(model_path) == ()

        sidecar = Path(tmp_dir) / "model.onnx.data"
        sidecar.write_bytes(b"data")
        assert find_external_data_files(model_path) == ("model.onnx.data",)
