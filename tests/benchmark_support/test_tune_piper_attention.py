import pytest
from lib.tuning import TuningPhase
from tune_piper_attention import _parse_args, _validate_args


def test_tuner_defaults_to_hot_pointer_descriptor_search() -> None:
    arguments = _parse_args([])

    assert arguments.schedules == ["pointer", "tensor-descriptor"]
    assert arguments.block_m == [32, 64, 128]
    assert arguments.num_warps == [4, 8]
    assert arguments.num_stages == [2, 3, 4]
    assert arguments.mixed_sign == "native"
    assert arguments.phase is TuningPhase.PREPARED_EXECUTION
    assert arguments.center_value is None


def test_tuner_accepts_end_to_end_uncentered_search() -> None:
    arguments = _parse_args(
        [
            "--phase",
            "operator_end_to_end",
            "--no-center-value",
            "--mixed-sign",
            "both",
        ]
    )

    assert arguments.phase is TuningPhase.OPERATOR_END_TO_END
    assert arguments.center_value is False
    assert arguments.mixed_sign == "both"


def test_tuner_rejects_causal_cross_attention() -> None:
    arguments = _parse_args(["--causal", "--sequence", "128", "--kv-sequence", "256"])

    with pytest.raises(SystemExit, match="equal query and key/value lengths"):
        _validate_args(arguments)
