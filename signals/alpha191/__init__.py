"""
Alpha 191 因子库。

使用:
    from signals.alpha191 import factor_001, factor_191, compute_factor_matrix
    from signals.alpha191.factors import factor_001  # 单个因子
"""

from .factors import (
    factor_001, factor_002, factor_003, factor_004, factor_005,
    factor_006, factor_007, factor_008, factor_009, factor_010,
    factor_011, factor_012, factor_013, factor_014, factor_015,
    factor_016, factor_017, factor_018, factor_019, factor_020,
    factor_021, factor_022, factor_023, factor_024, factor_025,
    factor_026, factor_027, factor_028, factor_029, factor_030,
    factor_031, factor_032, factor_033, factor_034, factor_035,
    factor_036, factor_037, factor_038, factor_039, factor_040,
    factor_041, factor_042, factor_043, factor_044, factor_045,
    factor_046, factor_047, factor_048, factor_049, factor_050,
    factor_051, factor_052, factor_053, factor_054, factor_055,
    factor_056, factor_057, factor_058, factor_059, factor_060,
    factor_061, factor_062, factor_063, factor_064, factor_065,
    factor_066, factor_067, factor_068, factor_069, factor_070,
    factor_071, factor_072, factor_073, factor_074, factor_075,
    factor_076, factor_077, factor_078, factor_079, factor_080,
    factor_081, factor_082, factor_083, factor_084, factor_085,
    factor_086, factor_087, factor_088, factor_089, factor_090,
    factor_091, factor_092, factor_093, factor_094, factor_095,
    factor_096, factor_097, factor_098, factor_099, factor_100,
    factor_101, factor_102, factor_103, factor_104, factor_105,
    factor_106, factor_107, factor_108, factor_109, factor_110,
    factor_111, factor_112, factor_113, factor_114, factor_115,
    factor_116, factor_117, factor_118, factor_119, factor_120,
    factor_121, factor_122, factor_123, factor_124, factor_125,
    factor_126, factor_127, factor_128, factor_129, factor_130,
    factor_131, factor_132, factor_133, factor_134, factor_135,
    factor_136, factor_137, factor_138, factor_139, factor_140,
    factor_141, factor_142, factor_143, factor_144, factor_145,
    factor_146, factor_147, factor_148, factor_149, factor_150,
    factor_151, factor_152, factor_153, factor_154, factor_155,
    factor_156, factor_157, factor_158, factor_159, factor_160,
    factor_161, factor_162, factor_163, factor_164, factor_165,
    factor_166, factor_167, factor_168, factor_169, factor_170,
    factor_171, factor_172, factor_173, factor_174, factor_175,
    factor_176, factor_177, factor_178, factor_179, factor_180,
    factor_181, factor_182, factor_183, factor_184, factor_185,
    factor_186, factor_187, factor_188, factor_189, factor_190,
    factor_191,
)
from .calculator import compute_factor_matrix, list_factors, get_factor_func

__all__ = [
    "compute_factor_matrix",
    "list_factors",
] + [f"factor_{i:03d}" for i in range(1, 192)]
