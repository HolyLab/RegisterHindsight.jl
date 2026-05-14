"""
    RegisterHindsight

Hindsight-based image registration refinement.

This module provides a gradient-descent optimizer that refines a deformation
field `ϕ` by minimizing the mean-square intensity error between a `fixed` image
and a warped `moving` image, subject to a deformation penalty.

The primary entry point is [`optimize!`](@ref RegisterHindsight.optimize!). The
lower-level penalty functions (`penalty_hindsight`, `penalty_hindsight_data`,
`penalty_hindsight_reg`, and their in-place gradient variants) are also
accessible as `RegisterHindsight.penalty_hindsight` etc.

The deformation `ϕ` must be an [`InterpolatingDeformation`](@ref), i.e., a
`GridDeformation` whose displacement field is backed by a `ScaledInterpolation`
with `InPlace` boundary conditions. Construct one with `interpolate!(copy(ϕ0))`
from a plain `GridDeformation` `ϕ0` (note: `interpolate(ϕ0)` produces a
different, incompatible type).
"""
module RegisterHindsight

using ImageCore: ImageCore, float64, gray
using StaticArrays: StaticArrays
using OffsetArrays: OffsetArrays
using ProgressMeter: ProgressMeter, ProgressUnknown
using Interpolations: Interpolations, AbstractExtrapolation, AbstractInterpolation,
    BSpline, Linear, ScaledInterpolation
# Interpolations internals needed to manually evaluate weighted interpolation sums;
# workaround for Interpolations v0.15 removing WeightedArbIndex as a direct array index.
using Interpolations: tcollect, itpflag, value_weights, coefficients, indextuple, weights
using RegisterDeformation: RegisterDeformation, GridDeformation, extrapolate, interpolate
using RegisterPenalty: RegisterPenalty, DeformationPenalty, penalty!

# optimize! is deliberately unexported because it conflicts with RegisterOptimize.optimize!

"""
    InterpolatingDeformation{T, N, A <: ScaledInterpolation}

Type alias for a `GridDeformation` whose displacement field is backed by a
`ScaledInterpolation`. All functions in this module operate on deformations of
this type.

Construct one with `interpolate!(copy(ϕ0))` from a plain `GridDeformation`
`ϕ0`. Note that `interpolate(ϕ0)` (without `!`) produces a different,
incompatible type.

!!! note
    The regularization penalty (`penalty_hindsight_reg!`) applies to the
    interpolation coefficient array and therefore requires `InPlace` boundary
    conditions so that the coefficient array has no padding. If the deformation
    was not constructed with `InPlace` boundary conditions, a runtime error is
    thrown.
"""
const InterpolatingDeformation{T, N, A <: ScaledInterpolation} = GridDeformation{T, N, A}

# Interpolations v0.15 no longer accepts WeightedArbIndex as a direct array index.
# Compute the weighted sum over interpolation coefficients manually.
function _coefs_at(coefs, wI)
    idxs = map(indextuple, wI)
    ws = map(weights, wI)
    result = zero(eltype(coefs))
    for J in CartesianIndices(map(length, idxs))
        idx = CartesianIndex(map(getindex, idxs, Tuple(J)))
        w = prod(map(getindex, ws, Tuple(J)))
        @inbounds result += w * coefs[idx]
    end
    return result
end

"""
    penalty_hindsight(ϕ, dp, fixed, moving) → scalar

Return the total hindsight penalty for deformation `ϕ`: the sum of the data
penalty (mean-square intensity error) and the regularization penalty from `dp`.

`moving` must be an `AbstractInterpolation`.

See also [`penalty_hindsight_data`](@ref RegisterHindsight.penalty_hindsight_data),
[`penalty_hindsight_reg`](@ref RegisterHindsight.penalty_hindsight_reg),
[`penalty_hindsight!`](@ref RegisterHindsight.penalty_hindsight!).
"""
function penalty_hindsight(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving)
    return penalty_hindsight_reg(ϕ, dp) + penalty_hindsight_data(ϕ, fixed, moving)
end

"""
    penalty_hindsight!(g, ϕ, dp, fixed, moving) → scalar

Compute the total hindsight penalty for `ϕ` and write its gradient to `g`.
Returns the same scalar value as
[`penalty_hindsight`](@ref RegisterHindsight.penalty_hindsight)`(ϕ, dp, fixed, moving)`.

`g` must be allocated as `similar(ϕ.u.itp.coefs)` and is written in-place
(replacing any previous contents).
"""
function penalty_hindsight!(g, ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving)
    gd = similar(g)
    ret = penalty_hindsight_reg!(g, ϕ, dp) + penalty_hindsight_data!(gd, ϕ, fixed, moving)
    for i in eachindex(g)
        g[i] += gd[i]
    end
    return ret
end

"""
    penalty_hindsight(ϕ1, ϕ2, dp, fixed, moving) → (scalar, scalar)

Return the total hindsight penalties for two deformations `ϕ1` and `ϕ2` as a
tuple `(p1, p2)`, evaluated on the same set of voxels. A voxel contributes to
both penalties only if both `ϕ1(x)` and `ϕ2(x)` map to in-bounds (finite)
positions in `moving`, ensuring the two values are directly comparable.

`ϕ1` and `ϕ2` must have the same `axes` and `nodes`.
"""
function penalty_hindsight(
        ϕ1::InterpolatingDeformation,
        ϕ2::InterpolatingDeformation,
        dp::DeformationPenalty, fixed, moving
    )
    axes(ϕ1.u) == axes(ϕ2.u) || throw(DimensionMismatch("The axes of the two deformations must match, got $(axes(ϕ1.u)) and $(axes(ϕ2.u))"))
    rp1, rp2 = penalty_hindsight_reg(ϕ1, dp), penalty_hindsight_reg(ϕ2, dp)
    dp1, dp2 = penalty_hindsight_data(ϕ1, ϕ2, fixed, moving)
    return rp1 + dp1, rp2 + dp2
end

"""
    penalty_hindsight_reg(ϕ, dp) → scalar

Return the regularization component of the hindsight penalty for deformation
`ϕ`, computed by `RegisterPenalty.penalty!` applied to the interpolation
coefficient array of `ϕ`.

!!! note
    Requires `InPlace` boundary conditions. See [`InterpolatingDeformation`](@ref).
"""
function penalty_hindsight_reg(ϕ::InterpolatingDeformation, dp)
    return penalty_hindsight_reg!(nothing, ϕ, dp)
end

"""
    penalty_hindsight_reg!(g, ϕ, dp) → scalar

Compute the regularization penalty for `ϕ` and write its gradient to `g`.
Returns the same scalar value as
[`penalty_hindsight_reg`](@ref RegisterHindsight.penalty_hindsight_reg)`(ϕ, dp)`.

`g` must be `nothing` (to skip gradient computation) or allocated as
`similar(ϕ.u.itp.coefs)` and is written in-place.

Throws an error if the deformation does not use `InPlace` boundary conditions
(i.e., if `axes(ϕ.u.itp.coefs) ≠ ϕ.u.itp.parentaxes`).
"""
function penalty_hindsight_reg!(g, ϕ::InterpolatingDeformation, dp)
    # The regularization penalty. We apply this to the interpolation
    # coefficients rather than the on-grid values. This may be
    # cheating. It also requires InPlace() so that the sizes match.
    itp = ϕ.u.itp
    axes(itp.coefs) == itp.parentaxes || error("deformation cannot have padding (use `InPlace` boundary conditions)")
    return penalty!(g, dp, itp.coefs)
end

"""
    penalty_hindsight_data(ϕ, fixed, moving) → scalar

Return the data penalty for deformation `ϕ`: the mean-square intensity error
between `fixed` and `moving` evaluated at the deformed positions `ϕ(x)`,
averaged over all voxels where both `fixed[x]` and `moving(ϕ(x))` are finite.

`moving` must be an `AbstractInterpolation` (typically also an extrapolation
returning `NaN` for out-of-bounds coordinates).
"""
function penalty_hindsight_data(
        ϕ::InterpolatingDeformation{T, N, A},
        fixed::AbstractArray{T1, N},
        moving::AbstractInterpolation{T2, N}
    ) where {T, N, T1, T2, A}
    # precomputing the weights makes the loop more efficient
    coefs, windexes = prepare_value_axes(ϕ)
    valid = 0
    Tout = promote_type(T1, T2, eltype(eltype(A)))
    mm = zero(Tout)
    for (I, wI) in zip(CartesianIndices(fixed), Iterators.product(windexes...))
        fval = fixed[I]
        if isfinite(fval)
            offset = _coefs_at(coefs, wI)
            mval = moving((Tuple(I) .+ Tuple(offset))...)
            if isfinite(mval)
                valid += 1
                diff = Tout(fval) - Tout(mval)
                mm += diff^2
            end
        end
    end
    return mm / valid
end

"""
    penalty_hindsight_data!(g, ϕ, fixed, moving) → scalar

Compute the data penalty for `ϕ` and write its gradient with respect to the
interpolation coefficients of `ϕ` to `g`. Returns the same scalar value as
[`penalty_hindsight_data`](@ref RegisterHindsight.penalty_hindsight_data)`(ϕ, fixed, moving)`.

`g` must be allocated as `similar(ϕ.u.itp.coefs)` and is zeroed and written
in-place. Its element type must be an `SVector` matching the spatial
dimensionality of `ϕ` (e.g., `SVector{2, Float64}` for 2-D images).

`moving` must be an `AbstractInterpolation` (not merely an extrapolation) so
that `Interpolations.gradient` can be evaluated on it.
"""
# This re-uses the work of computing the weights for both the value and the gradient
function penalty_hindsight_data!(
        g,
        ϕ::InterpolatingDeformation{T, N},
        fixed::AbstractArray{T1, N},
        moving::AbstractInterpolation{T2, N}
    ) where {T, N, T1, T2}
    coefs, windexes = prepare_value_axes(ϕ)
    fill!(g, zero(eltype(g)))
    valid = 0
    mm = 0.0
    # The following is like
    #   for (I, wI) in zip(, Iterators.product(windexes...))
    # but a bit easier on inference & optimization
    for I in CartesianIndices(fixed)
        @inbounds fval = fixed[I]
        if isfinite(fval)
            wI = map(getindex, windexes, Tuple(I))
            offset = _coefs_at(coefs, wI)
            ϕxindexes = Tuple(I) .+ Tuple(offset)
            mval = moving(ϕxindexes...)
            if isfinite(mval)
                valid += 1
                diff = gray(float64(fval) - float64(mval))
                mm += abs2(diff)
                # For the elements of the gradient we use the
                # chain rule, and thus need the spatial gradient
                # of the image
                gimg = (-2 * diff) * Interpolations.gradient(moving.itp, ϕxindexes...)
                # The following is like
                #   for (idx, w) in zip(Iterators.product(map(indextuple, wI)...), Iterators.product(map(weights, wI)...))
                # but a bit easier on inference & optimization
                idxs, ws = map(indextuple, wI), map(weights, wI)
                for J in CartesianIndices(map(length, idxs))
                    idx = CartesianIndex(map(getindex, idxs, Tuple(J)))
                    w = map(getindex, ws, Tuple(J))
                    @inbounds g[idx] += prod(w) * gimg
                end
            end
        end
    end
    for i in eachindex(g)
        g[i] /= valid
    end
    return mm / valid
end

"""
    coefs, wis = prepare_value_axes(ϕ::InterpolatingDeformation)

Return the interpolation coefficient array `coefs` and a tuple `wis` of
per-dimension `WeightedIndex` arrays for evaluating `ϕ` at each position in
its node grid. `wis[d][i]` is the `WeightedIndex` for the `i`-th position
along dimension `d`.

This is a low-level helper used internally by
[`penalty_hindsight_data`](@ref RegisterHindsight.penalty_hindsight_data) and
[`penalty_hindsight_data!`](@ref RegisterHindsight.penalty_hindsight_data!).
The coefficient-weighted sum at a given node position is computed by `_coefs_at`.
"""
function prepare_value_axes(ϕ::InterpolatingDeformation)
    itp = ϕ.u.itp
    itpflags = tcollect(itpflag, itp)
    nodes = ϕ.nodes
    newaxes = map(r -> Base.Slice(round(Int, first(r)):round(Int, last(r))), nodes)
    wis = Interpolations.dimension_wis(value_weights, itpflags, axes(itp), newaxes, nodes)
    return coefficients(itp), wis
end

"""
    penalty_hindsight_data(ϕ1, ϕ2, fixed, moving) → (scalar, scalar)

Return the data penalties for two deformations `ϕ1` and `ϕ2` as a tuple
`(p1, p2)`, evaluated on the same set of voxels. A voxel contributes to both
penalties only if `moving` returns a finite value at both `ϕ1(x)` and
`ϕ2(x)`, ensuring the two values are directly comparable.

`ϕ1` and `ϕ2` must have the same `nodes`.
"""
# This implementation allows comparing ϕ1 and ϕ2 on equal footing, meaning using the same
# voxels of the image data. A voxel gets included only if both ϕ1 and ϕ2 are in-bounds.
# This cannot be guaranteed for the single-ϕ implementation of penalty_hindsight_data.
function penalty_hindsight_data(
        ϕ1::InterpolatingDeformation{T, N},
        ϕ2::InterpolatingDeformation{T, N},
        fixed::AbstractArray{T1, N},
        moving::AbstractInterpolation{T2, N}
    ) where {T, N, T1, T2}
    coefs1, windexes1 = prepare_value_axes(ϕ1)
    coefs2, windexes2 = prepare_value_axes(ϕ2)
    nodes = ϕ1.nodes
    ϕ2.nodes == nodes || error("nodes of ϕ1 and ϕ2 must be the same, got $nodes and $(ϕ2.nodes), respectively")
    valid = 0
    mm1 = mm2 = 0.0
    for I in CartesianIndices(fixed)
        @inbounds fval = fixed[I]
        if isfinite(fval)
            wI1 = map(getindex, windexes1, Tuple(I))
            wI2 = map(getindex, windexes2, Tuple(I))
            offset1, offset2 = _coefs_at(coefs1, wI1), _coefs_at(coefs2, wI2)
            mval1 = moving((Tuple(I) .+ Tuple(offset1))...)
            mval2 = moving((Tuple(I) .+ Tuple(offset2))...)
            if isfinite(mval1) && isfinite(mval2)
                valid += 1
                diff = float64(fval) - float64(mval1)
                mm1 += diff^2
                diff = float64(fval) - float64(mval2)
                mm2 += diff^2
            end
        end
    end
    return mm1 / valid, mm2 / valid
end

"""
    result = optimize!(ϕ, dp, fixed, moving; stepsize=1.0, itermax=1000)

Minimize the mean-square error between `fixed[x...]` and `moving(ϕ(x)...)` by
updating `ϕ` in place via gradient descent on the interpolation coefficients.
`dp` is a deformation penalty (e.g., `AffinePenalty`). `stepsize` is the
maximum per-iteration update in pixels applied to elements of `ϕ`.

`moving` may be:
- an `AbstractExtrapolation` (used directly),
- an `AbstractInterpolation` (wrapped automatically with `NaN` extrapolation), or
- a plain `AbstractArray` (interpolated with `BSpline(Linear())` then extrapolated).

Terminates when a gradient-descent step would increase the penalty value, or
after `itermax` iterations.

# Returns

A named tuple `(; final, initial)` where `final` is the penalty after
optimization and `initial` is the penalty before optimization. Tuple
destructuring also works: `final, initial = optimize!(...)`.

# Example

```jldoctest
julia> using RegisterDeformation, RegisterPenalty, Interpolations

julia> fixed = sin.(range(0, stop=4π, length=40));

julia> nodes = (range(1, stop=40, length=5),);

julia> ϕ = interpolate!(GridDeformation(zeros(1, 5), nodes));

julia> ap = AffinePenalty(ϕ.nodes, 0.01);

julia> moving = sin.(range(0.2, stop=4π + 0.2, length=40));

julia> result = RegisterHindsight.optimize!(ϕ, ap, fixed, moving; stepsize=0.1);

julia> result.final < result.initial
true
```
"""
function optimize!(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving::AbstractExtrapolation; stepsize = 1.0, itermax = 1000)
    # Optimize the interpolation coefficients, rather than the values
    # of the deformation at the grid points
    ϕtrial = deepcopy(ϕ)
    g = similar(ϕ.u.itp.coefs)
    objective = ϕ -> penalty_hindsight(ϕ, dp, fixed, moving)
    objective2 = (ϕ1, ϕ2) -> penalty_hindsight(ϕ1, ϕ2, dp, fixed, moving)
    ∇objective!(g, ϕ) = penalty_hindsight!(g, ϕ, dp, fixed, moving)
    pold = p0 = objective(ϕ)
    iter = 0
    prog = ProgressUnknown(; desc="Performing descent:")
    while iter < itermax
        iter += 1
        ∇objective!(g, ϕ)
        gmax = mapreduce(v -> maximum(abs, v), max, g)
        if gmax == 0 || !isfinite(gmax)
            break
        end
        s = eltype(eltype(g))(stepsize / gmax)
        copyto!(ϕtrial.u.itp.coefs, ϕ.u.itp.coefs .- s .* g)
        p, pold = objective2(ϕtrial, ϕ)
        if p >= pold
            ProgressMeter.finish!(prog)
            break
        end
        copyto!(ϕ.u.itp.coefs, ϕtrial.u.itp.coefs)
        ProgressMeter.next!(prog; showvalues = [(:penalty, p)])
    end
    return (; final = pold, initial = p0)
end

function optimize!(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving::AbstractInterpolation; kwargs...)
    emoving = extrapolate(moving, NaN)
    return optimize!(ϕ, dp, fixed, emoving; kwargs...)
end

function optimize!(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving::AbstractArray; kwargs...)
    # imoving = interpolate(moving, BSpline(Quadratic(Flat(OnCell()))))
    imoving = interpolate(moving, BSpline(Linear()))
    return optimize!(ϕ, dp, fixed, imoving; kwargs...)
end

end
