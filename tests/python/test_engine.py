from oaa import Engine


def test_python_to_cpp_smoke() -> None:
    engine = Engine()

    assert engine.status() == "empty"
    assert engine.load_model("phase1-placeholder") is True
    assert engine.loaded is True
    assert engine.generate("hello") == "OAA_PHASE1_ECHO: hello"

    stats = engine.get_stats()
    assert stats == {
        "model_loaded": True,
        "generation_calls": 1,
        "model_path": "phase1-placeholder",
    }

    engine.unload()
    assert engine.loaded is False
