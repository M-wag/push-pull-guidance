import random
import pytest
from mylib.diffusion import Config
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigTest(Config):
    x: Any | list[Any]
    y: Any | list[Any]
    z: Any | list[Any]

@dataclass(frozen=True)
class ConfigNested(Config):
    a: Any | list[Any]
    b: Any | list[Any]


def test_config_split_matches_cartesian_product():
    len_x = random.randint(0, 5)
    len_y = random.randint(0, 5)
    len_z = random.randint(0, 5)

    xs = [random.randint(0, 99) for _ in range(len_x)]
    ys = [random.randint(0, 99) for _ in range(len_y)]
    zs = [random.randint(0, 99) for _ in range(len_z)]

    cnfg = ConfigTest(x=xs, y=ys, z=zs)
    split_cfgs = cnfg.split()

    expected_count = len_x * len_y * len_z
    assert len(split_cfgs) == expected_count, (
        f"Expected {expected_count} split configs, "
        f"got {len(split_cfgs)} (len_x={len_x}, len_y={len_y}, len_z={len_z})"
    )

    manual_cfgs = [
        ConfigTest(x=x, y=y, z=z)
        for x in xs
        for y in ys
        for z in zs
    ]

    for idx, (auto, manual) in enumerate(zip(split_cfgs, manual_cfgs)):
        if auto != manual:
            pytest.fail(
                f"Mismatch at index {idx}:\n"
                f"  auto:   {auto!r}\n"
                f"  manual: {manual!r}"
            )



def test_config_split_with_no_lists_returns_self():
    cnfg = ConfigTest(x=0, y=1, z=2)
    assert cnfg.split() == [cnfg], (
        "split() should return [self] when all values are scalars.\n"
        f"split() output: {cnfg.split()}\n"
    )

def test_config_split_is_recursive():
    cnfg = ConfigTest(
            x=ConfigNested(a=1, b=[2,3]),
            y=1, 
            z=2
    )

    split_cnfgs = cnfg.split()
    manual_cnfgs = [
            ConfigTest(
                x=ConfigNested(a=1, b=2),
                y=1,
                z=2
                ),
            ConfigTest(
                x=ConfigNested(a=1, b=3),
                y=1,
                z=2
                ),
            ]

    for idx, (auto, manual) in enumerate(zip(split_cnfgs, manual_cnfgs)):
        if auto != manual:
            pytest.fail(
                f"Mismatch at index {idx}:\n"
                f"  auto:   {auto!r}\n"
                f"  manual: {manual!r}"
            )

def test_config_shape_combination():
    cnfg = ConfigTest(x=[1, 2, 3], y=[10, 20], z=[0])
    assert cnfg.shape_combination == (3, 2, 1), (
        f"Expected shape_combination (3, 2, 1), got {cnfg.shape_combination}"
    )

