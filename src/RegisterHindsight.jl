module RegisterHindsight

using ImageCore, StaticArrays, OffsetArrays, ProgressMeter
using Interpolations, RegisterDeformation, RegisterPenalty
using Interpolations: tcollect, itpflag, value_weights, coefficients, indextuple, weights

# optimize! is deliberately unexported because it conflicts with RegisterOptimize.optimize!

const InterpolatingDeformation{T,N,A<:ScaledInterpolation} = GridDeformation{T,N,A}

function penalty_hindsight(ϕ::InterpolatingDeformation, ap::AffinePenalty, fixed, moving)
    return penalty_hindsight_reg(ap, ϕ) + penalty_hindsight_data(ϕ, fixed, moving)
end

function penalty_hindsight!(g, ϕ::InterpolatingDeformation, ap::AffinePenalty, fixed, moving)
    gd = similar(g)
    ret = penalty_hindsight_reg!(g, ap, ϕ) + penalty_hindsight_data!(gd, ϕ, fixed, moving)
    for i in eachindex(g)
        g[i] += gd[i]
    end
    return ret
end

# For comparison of two deformations
function penalty_hindsight(ϕ1::InterpolatingDeformation,
                           ϕ2::InterpolatingDeformation,
                           ap::AffinePenalty, fixed, moving)
    axes(ϕ1.u) == axes(ϕ2.u) || throw(DimensionMismatch("The axes of the two deformations must match, got $(axes(U1)) and $(axes(U2))"))
    rp1, rp2 = penalty_hindsight_reg(ap, ϕ1), penalty_hindsight_reg(ap, ϕ2)
    dp1, dp2 = penalty_hindsight_data(ϕ1, ϕ2, fixed, moving)
    rp1+dp1, rp2+dp2
end

function penalty_hindsight_reg(ap, ϕ::InterpolatingDeformation)
    penalty_hindsight_reg!(nothing, ap, ϕ)
end

function penalty_hindsight_reg!(g, ap, ϕ::InterpolatingDeformation)
    # The regularization penalty. We apply this to the interpolation
    # coefficients rather than the on-grid values. This may be
    # cheating. It also requires InPlace() so that the sizes match.
    itp = ϕ.u.itp
    axes(itp.coefs) == itp.parentaxes || error("deformation cannot have padding (use `InPlace` boundary conditions)")
    penalty!(g, ap, itp.coefs)
end

function penalty_hindsight_data(ϕ::InterpolatingDeformation{T,N,A},
                                fixed::AbstractArray{T1,N},
                                moving::AbstractInterpolation{T2,N}) where {T,N,T1,T2,A}
    # precomputing the weights makes the loop more efficient
    coefs, windexes = prepare_value_axes(ϕ)
    valid = 0
    Tout = promote_type(T1, T2, eltype(eltype(A)))
    mm = zero(Tout)
    for (I, wI) in zip(CartesianIndices(fixed), Iterators.product(windexes...))
        fval = fixed[I]
        if isfinite(fval)
            @inbounds offset = coefs[wI...]
            mval = moving((Tuple(I) .+ Tuple(offset))...)
            if isfinite(mval)
                valid += 1
                diff = Tout(fval)-Tout(mval)
                mm += diff^2
            end
        end
    end
    return mm/valid
end

# This re-uses the work of computing the weights for both the value and the gradient
function penalty_hindsight_data!(g,
                                 ϕ::InterpolatingDeformation{T,N},
                                 fixed::AbstractArray{T1,N},
                                 moving::AbstractInterpolation{T2,N}) where {T,N,T1,T2}
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
            @inbounds offset = coefs[wI...]
            ϕxindexes = Tuple(I) .+ Tuple(offset)
            mval = moving(ϕxindexes...)
            if isfinite(mval)
                valid += 1
                diff = gray(float64(fval)-float64(mval))
                mm += abs2(diff)
                # For the elements of the gradient we use the
                # chain rule, and thus need the spatial gradient
                # of the image
                gimg = (-2*diff)*Interpolations.gradient(moving.itp, ϕxindexes...)
                # The following is like
                #   for (idx, w) in zip(Iterators.product(map(indextuple, wI)...), Iterators.product(map(weights, wI)...))
                # but a bit easier on inference & optimization
                idxs, ws = map(indextuple, wI), map(weights, wI)
                for J in CartesianIndices(map(length, idxs))
                    idx = CartesianIndex(map(getindex, idxs, Tuple(J)))
                    w = map(getindex, ws, Tuple(J))
                    @inbounds g[idx] += prod(w)*gimg
                end
            end
        end
    end
    for i in eachindex(g)
        g[i] /= valid
    end
    return mm/valid
end

"""
    coefs, wI = prepare_value_axes(ϕ::InterpolatingDeformation)

Return the coefficients and `Interpolations.WeightedIndex` values needed for evaluating
`ϕ` at each position in the range grid. `coefs` is an array and `wI` a tuple, `wI[d]`
corresponding to axis `d` of the range grid. In particular, at location `I = (I1, ..., In)`
the value is `coefs[wI[1][I1], ..., wI[n][In]]`.
"""
function prepare_value_axes(ϕ::InterpolatingDeformation)
    itp = ϕ.u.itp
    itpflags = tcollect(itpflag, itp)
    nodes = ϕ.nodes
    newaxes = map(r->Base.Slice(round(Int, first(r)):round(Int, last(r))), nodes)
    wis = Interpolations.dimension_wis(value_weights, itpflags, axes(itp), newaxes, nodes)
    return coefficients(itp), wis
end

# This implementation allows comparing ϕ1 and ϕ2 on equal footing, meaning using the same
# voxels of the image data. A voxel gets included only if both ϕ1 and ϕ2 are in-bounds.
# This cannot be guaranteed for the single-ϕ implementation of penalty_hindsight_data.
function penalty_hindsight_data(ϕ1::InterpolatingDeformation{T,N},
                                ϕ2::InterpolatingDeformation{T,N},
                                fixed::AbstractArray{T1,N},
                                moving::AbstractInterpolation{T2,N}) where {T,N,T1,T2}
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
            offset1, offset2 = coefs1[wI1...], coefs2[wI2...]
            mval1 = moving((Tuple(I) .+ Tuple(offset1))...)
            mval2 = moving((Tuple(I) .+ Tuple(offset2))...)
            if isfinite(mval1) && isfinite(mval2)
                valid += 1
                diff = float64(fval)-float64(mval1)
                mm1 += diff^2
                diff = float64(fval)-float64(mval2)
                mm2 += diff^2
            end
        end
    end
    return mm1/valid, mm2/valid
end

"""
    pnew, pold = optimize!(ϕ, dp, fixed, moving; stepsize=1, itermax=1000)

Improve `ϕ` by minimizing the mean-square error between `fixed[x...]`
and `moving(ϕ(x)...)`. `dp` is a deformation penalty, e.g.,
an `AffinePenalty`. `stepsize` specifies the maximum update, in pixels, of
elements of `ϕ`.

The operation terminates when the step increases the value of the penalty,
or when more than `itermax` iterations have occurred.
"""
function optimize!(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving::AbstractExtrapolation; stepsize = 1.0, itermax=1000)
    # Optimize the interpolation coefficients, rather than the values
    # of the deformation at the grid points
    ϕtrial = deepcopy(ϕ)
    g = similar(ϕ.u.itp.coefs)
    objective = ϕ->penalty_hindsight(ϕ, dp, fixed, moving)
    objective2 = (ϕ1,ϕ2)->penalty_hindsight(ϕ1, ϕ2, dp, fixed, moving)
    ∇objective!(g, ϕ) = penalty_hindsight!(g, ϕ, dp, fixed, moving)
    pold = p0 = objective(ϕ)
    iter = 0
    prog = ProgressUnknown("Performing descent:")
    while iter < itermax
        iter += 1
        ∇objective!(g, ϕ)
        gmax = mapreduce(v->maximum(abs, v), max, g)
        if gmax == 0 || !isfinite(gmax)
            break
        end
        s = eltype(eltype(g))(stepsize/gmax)
        copyto!(ϕtrial.u.itp.coefs, ϕ.u.itp.coefs .- s .* g)
        p, pold = objective2(ϕtrial, ϕ)
        if p >= pold
            ProgressMeter.finish!(prog)
            break
        end
        copyto!(ϕ.u.itp.coefs, ϕtrial.u.itp.coefs)
        ProgressMeter.next!(prog; showvalues = [(:penalty,p)])
    end
    pold, p0
end

function optimize!(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving::AbstractInterpolation; kwargs...)
    emoving = extrapolate(moving, NaN)
    optimize!(ϕ, dp, fixed, emoving; kwargs...)
end

function optimize!(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving::AbstractArray; kwargs...)
    # imoving = interpolate(moving, BSpline(Quadratic(Flat(OnCell()))))
    imoving = interpolate(moving, BSpline(Linear()))
    optimize!(ϕ, dp, fixed, imoving; kwargs...)
end

end
