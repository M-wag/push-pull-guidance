import random
import pytest
from mylib.diffusion import Config
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigTest(Config):
    x: list[Any]
    y: list[Any]
    z: list[Any]

def test_Config_splits():
    # Init lists of random lengths 
    len_a = random.randint(1, 5)
    len_b = random.randint(1, 5)
    len_c = random.randint(1, 5)
    a = [random.randint(0, 99) for _ in range(len_a)]
    b = [random.randint(0, 99) for _ in range(len_b)]
    c = [random.randint(0, 99) for _ in range(len_c)]
    cnfg = ConfigTest(x=a, y=b, z=c)      

    # Check count
    cnfgs_split = cnfg.split()
    expected_count = len_a * len_b * len_c
    assert len(cnfgs_split) == expected_count, (
        f"Expected {expected_count} split configs, "
        f"got {len(cnfgs_split)} (len(a)={len_a}, len(b)={len_b}, len(c)={len_c})"
    )

    # Manually build all combinations of parameters
    cnfgs_manual = [ConfigTest(x=xi, y=yi, z=zi) 
                    for xi in a 
                    for yi in b 
                    for zi in c]

    # Compare element-wise and report first mismatch
    for idx, (auto, manual) in enumerate(zip(cnfgs_split, cnfgs_manual)):
        if auto != manual:
            pytest.fail(
                f"Mismatch at index {idx}:\n"
                f"  auto:   {auto!r}\n"
                f"  manual: {manual!r}"
            )

    assert cnfgs_split == cnfgs_manual

