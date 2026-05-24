# SPDX-License-Identifier: Apache-2.0

from .flash_attn_interface import flash_attn_varlen_func  # noqa: F401

# Import _C to register torch ops (dynamic_per_token_quant_int8_asym etc.)
from . import _C  # noqa: F401
