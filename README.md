# RegisterHindsight

[![CI](https://github.com/HolyLab/RegisterHindsight.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterHindsight.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/RegisterHindsight.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterHindsight.jl)

RegisterHindsight refines image-registration deformation fields using gradient
descent. Given a `fixed` image and a `moving` image, it adjusts the
displacements of a `GridDeformation` to minimize the mean-square intensity
error subject to a deformation-smoothness penalty.

It is part of the
[HolyLab registration pipeline](https://github.com/HolyLab/HolyLabRegistry.git)
and is designed to be called after a coarse registration step (e.g.,
[RegisterOptimize](https://github.com/HolyLab/RegisterOptimize.jl)) to squeeze
out residual misalignment.

## Installation

RegisterHindsight is registered in the
[HolyLab registry](https://github.com/HolyLab/HolyLabRegistry). Add the
registry once, then install normally:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterHindsight")
```

## Concept

"Hindsight" refers to the optimization strategy: rather than working with the
raw displacement values on the deformation grid, the optimizer adjusts the
*interpolation coefficients* that back the deformation field. This exposes a
smooth, differentiable objective that can be descended with a simple line
search.

The deformation `ϕ` must be an **interpolating deformation** — a
`GridDeformation` whose displacement field is backed by a
`ScaledInterpolation` with `InPlace` boundary conditions. Construct one with:

```julia
ϕ = interpolate!(copy(ϕ0))   # note: interpolate! (with !), not interpolate
```

`interpolate(ϕ0)` (without `!`) produces a different, incompatible type.

## Usage

```julia
using RegisterDeformation, RegisterPenalty, Interpolations
using RegisterHindsight

# 1-D example: shift a sine wave by ~0.05 radians
fixed  = sin.(range(0, stop=4π, length=40))
moving = sin.(range(0.2, stop=4π + 0.2, length=40))

nodes = (range(1, stop=40, length=5),)
ϕ = interpolate!(GridDeformation(zeros(1, 5), nodes))

ap = AffinePenalty(ϕ.nodes, 0.01)   # smoothness penalty weight 0.01

result = RegisterHindsight.optimize!(ϕ, ap, fixed, moving; stepsize=0.1)

result.final < result.initial   # true — penalty decreased
```

`optimize!` returns a named tuple `(; final, initial)` with the penalty
before and after optimization, so you can check how much improvement was
achieved.

### Penalty functions

The penalty and its gradient can also be called directly, which is useful for
diagnostics or for building a custom optimizer:

| Function | Description |
|---|---|
| `RegisterHindsight.penalty_hindsight(ϕ, dp, fixed, moving)` | Total penalty (data + regularization) |
| `RegisterHindsight.penalty_hindsight!(g, ϕ, dp, fixed, moving)` | Total penalty + gradient in `g` |
| `RegisterHindsight.penalty_hindsight_data(ϕ, fixed, moving)` | Data (intensity error) term only |
| `RegisterHindsight.penalty_hindsight_reg(ϕ, dp)` | Regularization term only |

A two-deformation overload `penalty_hindsight(ϕ1, ϕ2, dp, fixed, moving)`
evaluates both deformations on the same set of valid voxels so the two
penalties are directly comparable.

> **Note:** `optimize!` is intentionally not exported because it would
> conflict with `RegisterOptimize.optimize!`. Call it as
> `RegisterHindsight.optimize!(...)`.
