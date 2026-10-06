from pathlib import Path

from oaa import Engine, GenerationConfig


def write_manifest(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "architecture=tiny_transformer_v1",
                "vocab_size=95",
                "hidden_size=16",
                "num_layers=1",
                "num_heads=4",
                "intermediate_size=32",
                "context_length=32",
                "seed=7",
            ]
        ),
        encoding="utf-8",
    )


def test_stream_is_token_granular(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    engine = Engine()
    engine.load_model(str(manifest))

    chunks = list(
        engine.generate_stream(
            "Hi",
            GenerationConfig(max_tokens=4, temperature=0.0),
        )
    )

    assert len(chunks) == 4
    assert all(len(chunk) == 1 for chunk in chunks)


def test_generation_respects_context(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    engine = Engine()
    engine.load_model(str(manifest))

    output = engine.generate(
        "Hello",
        GenerationConfig(max_tokens=100, temperature=0.0),
    )

    assert len(output) == 27
