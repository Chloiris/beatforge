from beatforge_api.export_safety import public_model_identifier, sanitize_export_metadata


def test_public_model_identifier_preserves_remote_id_and_strips_native_paths() -> None:
    assert public_model_identifier("Qwen/Qwen3-ForcedAligner-0.6B") == (
        "Qwen/Qwen3-ForcedAligner-0.6B"
    )
    assert public_model_identifier("/opt/models/Qwen3-ForcedAligner-0.6B") == (
        "Qwen3-ForcedAligner-0.6B"
    )
    assert public_model_identifier(r"C:\models\Qwen3-ForcedAligner-0.6B") == (
        "Qwen3-ForcedAligner-0.6B"
    )


def test_export_metadata_recurses_without_changing_non_path_values() -> None:
    payload = sanitize_export_metadata(
        {
            "modelPath": "/home/example/models/model.bin",
            "values": [1, "model/vendor-id", True],
        }
    )

    assert payload == {
        "modelPath": "<local-path>/model.bin",
        "values": [1, "model/vendor-id", True],
    }
