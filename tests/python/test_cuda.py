from oaa import cuda_available, cuda_compiled, cuda_vector_add


def test_cuda_backend_status_is_boolean() -> None:
    assert isinstance(cuda_compiled(), bool)
    assert isinstance(cuda_available(), bool)

    if not cuda_compiled():
        assert cuda_available() is False


def test_cuda_vector_add_when_device_is_available() -> None:
    if not cuda_available():
        return

    assert cuda_vector_add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
