from pathlib import Path

import pytest

from oaa import Engine, GenerationConfig


def write_manifest(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "architecture=tiny_transformer_v1",
                "vocab_size=95",
                "hidden_size=16",
                "num_layers=2",
                "num_heads=4",
                "intermediate_size=32",
                "context_length=64",
                "seed=42",
            ]
        ),
        encoding="utf-8",
    )


def test_generation_config_validation() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(max_tokens=0)
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-1.0)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=0.0)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.1)
    with pytest.raises(ValueError):
        GenerationConfig(repetition_penalty=0.5)


def test_real_transformer_generation(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    engine = Engine()
    assert engine.load_model(str(manifest))

    config = GenerationConfig(max_tokens=8, temperature=0.0, seed=123)
    output = engine.generate("hello", config)

    assert len(output) == 8
    assert all(32 <= ord(char) <= 126 for char in output)

    stats = engine.get_stats()
    assert stats["prompt_tokens"] == 5
    assert stats["generated_tokens"] == 8
    assert stats["generation_calls"] == 1
    assert stats["tokens_per_second"] >= 0.0


def test_deterministic_seeded_generation(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    config = GenerationConfig(
        max_tokens=12,
        temperature=0.8,
        top_k=10,
        top_p=0.95,
        repetition_penalty=1.1,
        seed=999,
    )

    first = Engine()
    second = Engine()
    first.load_model(str(manifest))
    second.load_model(str(manifest))

    assert first.generate("Hello", config) == second.generate(
        "Hello", config)


def test_sampling_controls(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    engine = Engine()
    engine.load_model(str(manifest))

    configs = [
        GenerationConfig(
            max_tokens=6,
            temperature=0.7,
            top_k=5,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=1,
        ),
        GenerationConfig(
            max_tokens=6,
            temperature=0.9,
            top_k=0,
            top_p=0.75,
            repetition_penalty=1.2,
            seed=2,
        ),
    ]

    for config in configs:
        output = engine.generate("Hello", config)
        assert len(output) == 6


def test_streaming_matches_generation(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    engine = Engine()
    engine.load_model(str(manifest))
    config = GenerationConfig(max_tokens=8, temperature=0.0, seed=123)

    direct = engine.generate("hello", config)
    streamed = "".join(engine.generate_stream("hello", config))

    assert streamed == direct


def test_model_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.manifest"
    manifest.write_text(
        "\n".join(
            [
                "architecture=unsupported",
                "vocab_size=95",
                "hidden_size=16",
                "num_layers=2",
                "num_heads=4",
            ]
        ),
        encoding="utf-8",
    )

    engine = Engine()
    with pytest.raises(ValueError):
        engine.load_model(str(manifest))


def test_python_input_validation() -> None:
    engine = Engine()

    with pytest.raises(ValueError):
        engine.load_model("")

    with pytest.raises(TypeError):
        engine.generate(123)  # type: ignore[arg-type]
