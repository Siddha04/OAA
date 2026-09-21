import pytest

from oaa import Engine, GenerationConfig


def test_generation_config_validation() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(max_tokens=0)

    with pytest.raises(ValueError):
        GenerationConfig(temperature=-1.0)


def test_python_to_cpp_configured_generation() -> None:
    engine = Engine()
    engine.load_model("phase3-placeholder")

    config = GenerationConfig(max_tokens=32, temperature=0.0)
    assert engine.generate("hello", config) == "OAA_PHASE1_ECHO: hello"


def test_python_to_cpp_streaming_boundary() -> None:
    engine = Engine()
    engine.load_model("phase3-placeholder")

    chunks = list(engine.generate_stream("hello"))
    assert "".join(chunks) == "OAA_PHASE1_ECHO: hello"
    assert all(chunks)


def test_python_input_validation() -> None:
    engine = Engine()

    with pytest.raises(ValueError):
        engine.load_model("")

    with pytest.raises(TypeError):
        engine.generate(123)  # type: ignore[arg-type]
