"""Backend-independent execution planning for Piper Attention."""

from dataclasses import asdict, dataclass

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)


@dataclass(frozen=True, slots=True)
class PiperAttentionExecutionPlan:
    """Host-side specialization and launch choices for one Piper invocation."""

    block_m: int
    grouped_qk: bool
    native_uint8: bool
    split_pv_head_dim: bool
    scaled_fp16_numerator: bool
    use_tensor_descriptors: bool
    num_warps: int = 4
    num_stages: int = 3
    reverse_causal_blocks: bool = False
    loop_num_stages: int | None = None
    loop_licm: bool = False
    use_packed_probability_conversion: bool = False
    use_sm89_d128_specialization: bool = False
    use_shared_value_scale: bool = False
    use_fused_kv_preprocessing: bool = False
    use_fp16_value_scale: bool = False
    derive_value_scale_multiplier: bool = False
    use_hybrid_fp32_fp16_numerator: bool = False
    round_probability_codes: bool = True

    def __post_init__(self) -> None:  # noqa: PLR0912 - plan invariants stay explicit
        if self.block_m not in BLOCK_M_VALUES:
            raise ValueError("Piper Attention block_m must be 32, 64, or 128")
        if self.num_warps not in NUM_WARPS_VALUES:
            raise ValueError("Piper Attention num_warps must be 2, 4, or 8")
        if self.num_stages not in NUM_STAGES_VALUES:
            raise ValueError("Piper Attention num_stages must be 1, 2, 3, or 4")
        if self.loop_num_stages not in LOOP_NUM_STAGES_VALUES:
            raise ValueError("Piper Attention loop_num_stages must be None, 1, 2, 3, or 4")
        if self.scaled_fp16_numerator and not self.split_pv_head_dim:
            raise ValueError("scaled FP16 numerator recurrence requires split PV")
        if self.use_packed_probability_conversion and not self.native_uint8:
            raise ValueError("packed probability conversion requires native UINT8 MMA")
        if self.use_sm89_d128_specialization and not self.native_uint8:
            raise ValueError("SM89 D128 specialization requires native UINT8 MMA")
        if self.use_sm89_d128_specialization and not self.split_pv_head_dim:
            raise ValueError("SM89 D128 specialization requires split PV")
        if self.use_sm89_d128_specialization and self.use_tensor_descriptors:
            raise ValueError("SM89 D128 specialization requires pointer loads")
        if self.use_shared_value_scale and not self.use_sm89_d128_specialization:
            raise ValueError("shared V scaling requires the SM89 D128 specialization")
        if self.use_fused_kv_preprocessing and not self.use_sm89_d128_specialization:
            raise ValueError("fused Q/K/V preprocessing requires the SM89 D128 specialization")
        if self.use_fp16_value_scale and not self.use_sm89_d128_specialization:
            raise ValueError("FP16 V-scale storage requires the SM89 D128 specialization")
        if self.use_fp16_value_scale and self.use_shared_value_scale:
            raise ValueError("FP16 per-key V-scale storage is incompatible with shared V scaling")
        if self.derive_value_scale_multiplier and not self.use_sm89_d128_specialization:
            raise ValueError("derived V-scale reconstruction requires the SM89 D128 specialization")
        if self.derive_value_scale_multiplier and not self.use_fused_kv_preprocessing:
            raise ValueError("derived V-scale reconstruction requires fused Q/K/V preprocessing")
        if self.derive_value_scale_multiplier and not self.use_fp16_value_scale:
            raise ValueError("derived V-scale reconstruction requires FP16 V-scale storage")
        if self.derive_value_scale_multiplier and self.use_shared_value_scale:
            raise ValueError("derived V-scale reconstruction requires per-key V scaling")
        if self.use_hybrid_fp32_fp16_numerator and not self.use_sm89_d128_specialization:
            raise ValueError("hybrid numerator accumulation requires the SM89 D128 specialization")
        if self.use_hybrid_fp32_fp16_numerator and self.scaled_fp16_numerator:
            raise ValueError("hybrid numerator accumulation requires the FP32 recurrence path")
        if self.use_hybrid_fp32_fp16_numerator and self.use_shared_value_scale:
            raise ValueError("hybrid numerator accumulation requires per-key V scaling")
        if not self.round_probability_codes and not self.use_sm89_d128_specialization:
            raise ValueError("probability truncation requires the SM89 D128 specialization")

    def as_dict(self) -> dict[str, object]:
        """Return execution choices as serializable benchmark metadata."""
        return asdict(self)


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    candidate_block_m: int,
    query_length: int,
    key_length: int,
    head_dim: int,
    is_causal: bool,
) -> PiperAttentionExecutionPlan:
    """Select established policy without borrowing schedules from other kernels."""
    grouped_qk = target.is_cuda_capability(12)
    native_uint8 = target.supports_uint8_int8_mma
    use_sm89_d128_specialization = (
        target.is_cuda_capability(8, 9)
        and not is_causal
        and head_dim == 128
        and query_length == key_length
        and query_length >= 8192
        and query_length % 128 == 0
        and key_length % 64 == 0
    )
    split_pv_head_dim = use_sm89_d128_specialization or (
        target.is_cuda_capability(12)
        and not is_causal
        and head_dim == 128
        and query_length >= 1024
        and key_length >= 1024
    )
    scaled_fp16_numerator = (
        split_pv_head_dim
        and key_length <= 131072
        and not (use_sm89_d128_specialization and key_length >= 131072)
    )
    # Paired SM120 measurements favor packed conversion for D64 and
    # non-causal D128, while the D128 causal specialization is neutral to
    # slightly slower and retains stock Triton lowering.
    use_packed_probability_conversion = use_sm89_d128_specialization or (
        target.is_cuda_capability(12, 0) and not (is_causal and head_dim == 128)
    )

    block_m = (
        128
        if use_sm89_d128_specialization
        else 64
        if is_causal
        else 128
        if scaled_fp16_numerator and query_length >= 8192 and key_length >= 8192
        else 64
        if split_pv_head_dim
        else candidate_block_m
    )
    use_tensor_descriptors = target.is_cuda_capability(12) and block_m == 128 and head_dim == 128
    return PiperAttentionExecutionPlan(
        block_m=block_m,
        grouped_qk=grouped_qk,
        native_uint8=native_uint8,
        split_pv_head_dim=split_pv_head_dim,
        scaled_fp16_numerator=scaled_fp16_numerator,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=(
            1
            if use_sm89_d128_specialization
            else 2
            if use_tensor_descriptors
            else 3
        ),
        loop_num_stages=(3 if use_sm89_d128_specialization else None),
        loop_licm=use_sm89_d128_specialization and key_length < 131072,
        use_packed_probability_conversion=use_packed_probability_conversion,
        use_sm89_d128_specialization=use_sm89_d128_specialization,
        # Per-key V scaling and probability rounding are production quality
        # gates. Shared-64-key scaling and truncation remain explicit offline
        # ablation axes, but both lose more than 0.5 dB on the SM89 corpus.
        use_shared_value_scale=False,
        use_fused_kv_preprocessing=use_sm89_d128_specialization,
        use_fp16_value_scale=use_sm89_d128_specialization,
        derive_value_scale_multiplier=(
            use_sm89_d128_specialization and key_length < 131072
        ),
        use_hybrid_fp32_fp16_numerator=(
            use_sm89_d128_specialization and key_length >= 131072
        ),
        round_probability_codes=True,
    )
