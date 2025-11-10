import torch
from pprint import pprint
from mylib.gvf_2 import BuilderPushPullVF, BuilderMap

def test_rank_2():
    B, D, L = 4, 2, 1
    map_args = {"seed": 0, "dim_in": D, "dim_out": L}
    builder = BuilderMap(map_args, None)
    map_fn, map_inv_fn = builder.build()
    
    x = torch.randn(B, D)
    z = map_fn(x)
    x_rec = map_inv_fn(z)
    assert z.shape == (B, L)
    assert x_rec.shape == (B, D)

def test_rank_2_channeled():
    B, D, L, C = 4, 2, 1, 3
    map_args = {"seed": 0, "dim_in": D, "dim_out": L, "n_features": C}
    builder = BuilderMap(map_args, None)
    map_fn, map_inv_fn = builder.build()
    
    x = torch.randn(B, D)
    z = map_fn(x)
    x_rec = map_inv_fn(z)
    assert z.shape == (B, C, L)
    assert x_rec.shape == (B, D)

def test_rank_3():
    B, D, L, N = 4, 2, 1, 5
    map_args = {"seed": 0, "dim_in": D, "dim_out": L}
    builder = BuilderMap(map_args, None)
    map_fn, map_inv_fn = builder.build()
    
    x = torch.randn(B, N, D)
    z = map_fn(x)
    x_rec = map_inv_fn(z)
    assert z.shape == (B, N, L)
    assert x_rec.shape == (B, N, D)

def test_rank_3_channeled():
    B, D, L, C, N = 4, 2, 1, 3, 5
    map_args = {"seed": 0, "dim_in": D, "dim_out": L, "n_features": C}
    builder = BuilderMap(map_args, None)
    map_fn, map_inv_fn = builder.build()
    
    x = torch.randn(B, N, D)
    z = map_fn(x)
    x_rec = map_inv_fn(z)
    assert z.shape == (B, N, C, L)
    assert x_rec.shape == (B, N, D)

