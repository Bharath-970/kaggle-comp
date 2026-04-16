from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostDeltaEstimate:
    extra_parameters: int
    extra_memory_bytes: int
    extra_macs: int

    @property
    def total(self) -> int:
        return self.extra_parameters + self.extra_memory_bytes + self.extra_macs


def estimate_conv_addition_cost(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    output_height: int = 30,
    output_width: int = 30,
    include_bias: bool = False,
) -> CostDeltaEstimate:
    kernel_params = out_channels * in_channels * kernel_size * kernel_size
    bias_params = out_channels if include_bias else 0
    parameters = kernel_params + bias_params
    memory_bytes = parameters * 4
    macs = out_channels * output_height * output_width * in_channels * kernel_size * kernel_size
    return CostDeltaEstimate(
        extra_parameters=parameters,
        extra_memory_bytes=memory_bytes,
        extra_macs=macs,
    )
