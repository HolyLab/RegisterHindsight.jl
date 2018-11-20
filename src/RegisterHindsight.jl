module RegisterHindsight

using Interpolations, RegisterDeformation, RegisterPenalty, ImageCore, StaticArrays, OffsetArrays
using Base.Cartesian
using Interpolations: sqr, SimpleRatio, BSplineInterpolation, DimSpec, Degree

const InterpolatingDeformation{T,N,A<:ScaledInterpolation} = GridDeformation{T,N,A}

function penalty_hindsight(ϕ::InterpolatingDeformation, ap::AffinePenalty{T}, fixed, moving) where T<:Real
    convert(T, penalty_hindsight_reg(ap, ϕ) +
               penalty_hindsight_data(ϕ, fixed, moving))
end

function penalty_hindsight!(g, ϕ::InterpolatingDeformation, ap::AffinePenalty{T}, fixed, moving) where T<:Real
    gd = similar(g)
    ret = convert(T, penalty_hindsight_reg!(g, ap, ϕ) +
                  penalty_hindsight_data!(gd, ϕ, fixed, moving))
    for i in eachindex(g)
        g[i] += gd[i]
    end
    ret
end

# For comparison of two deformations
function penalty_hindsight(ϕ1::InterpolatingDeformation,
                           ϕ2::InterpolatingDeformation,
                           ap::AffinePenalty{T}, fixed, moving) where T<:Real
    indices(ϕ1.u) == indices(ϕ2.u) || throw(DimensionMismatch("The indices of the two deformations must match, got $(indices(U1)) and $(indices(U2))"))
    rp1, rp2 = penalty_hindsight_reg(ap, ϕ1), penalty_hindsight_reg(ap, ϕ2)
    dp1, dp2 = penalty_hindsight_data(ϕ1, ϕ2, fixed, moving)
    convert(T, rp1+dp1), convert(T, rp2+dp2)
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

@generated function penalty_hindsight_data(ϕ::InterpolatingDeformation{T,N,A},
                                           fixed::AbstractArray{T1,N},
                                           moving::AbstractInterpolation{T2,N}) where {T,N,T1,T2,A}
    IT = Interpolations.itptype(A)
    uindexes = scaledindexes(IT, N)
    ϕxindexes = [:(I[$d] + u[$d]) for d = 1:N]
    out_type = promote_type(T1, T2, eltype(eltype(A)))
    quote
        knots = ϕ.knots
        steps = map(step, knots)
        offsets = map(first, knots)
        valid = 0
        mm = zero($(out_type))
        for I in CartesianRange(indices(fixed))
            fval = fixed[I]
            if isfinite(fval)
                u = ϕ.u.itp[$(uindexes...)]
                mval = moving[$(ϕxindexes...)]
                if isfinite(mval)
                    valid += 1
                    diff = $(out_type)(fval)-$(out_type)(mval)
                    mm += diff^2
                end
            end
        end
        mm/valid
    end
end

# To compute the derivative with respect to the deformation, it's
# efficient to re-use the coefficients computed for the shift. We do
# that by exploiting the generated code in Interpolations.
function penalty_hindsight_data!(g,
                                 ϕ::InterpolatingDeformation{T,N},
                                 fixed::AbstractArray{T1,N},
                                 moving::AbstractInterpolation{T2,N}) where {T,N,T1,T2}
    _penalty_hindsight_data!(g, ϕ.u.itp, ϕ.knots, fixed, moving)
end

@generated function _penalty_hindsight_data!(
    g,
    itp::BSplineInterpolation{T,N,TCoefs,IT,Axs},
    knots,
    fixed::AbstractArray{T1,N},
    moving::AbstractInterpolation{T2,N}) where {T,N,TCoefs,IT,Axs,T1,T2}

    penalty_hindsight_data!_gen(N, IT, Pad)
end

function penalty_hindsight_data!_gen(N, ::Type{IT}, Pad) where IT
    uindexes = scaledindexes(IT, N)
    xassign = Expr(:block, map((d,e)->Expr(:(=), Symbol("x_",d), e), 1:N, uindexes)...)
    ϕxindexes = [:(I[$d] + y[$d]) for d = 1:N]

    IR = interprange(IT, N)
    IA = OffsetArray(collect(CartesianIndices(IR)), IR)
    coef_exprs = [coef_gen(IT, 0, I) for I in IA]
    g_exprs = [:(g[$(map(Interpolations.offsetsym, I.I, 1:N)...)] += coef*$(coef_exprs[I])) for I in CartesianIndices(IR)]
    quote
        fill!(g, zero(eltype(g)))
        inds_itp = indices(itp)
        steps = map(step, knots)
        offsets = map(first, knots)
        valid = 0
        mm = 0.0
        gimg = gradient(moving.itp, map(first, indices(moving))...)
        GT = eltype(eltype(g))
        for I in CartesianIndices(indices(fixed))
            fval = fixed[I]
            if isfinite(fval)
                # This is effectively `y = ϕ.u.itp[$(uindexes...)]`,
                # except that we have local variables for the coefficients we can access
                $xassign
                $(Interpolations.define_indices(IT, N, Pad))
                $(Interpolations.coefficients(IT, N))
                @inbounds y = $(Interpolations.index_gen(IT, N))
                # End of `y = ϕ.u.itp[$(uindexes...)]`
                mval = moving[$(ϕxindexes...)]
                if isfinite(mval)
                    valid += 1
                    diff = float64(fval)-float64(mval)
                    mm += abs2(diff)
                    # For the elements of the gradient we use the
                    # chain rule, and thus need the spatial gradient
                    # of the image
                    gradient!(gimg, moving.itp, $(ϕxindexes...))
                    coef = (-2*diff)*SVector{$N,GT}(gimg)
                    $(Expr(:block, g_exprs...))
                end
            end
        end
        for i in eachindex(g)
            g[i] /= valid
        end
        mm/valid
    end
end

scaledindexes(::Type{IT}, N) where {IT} =
    N == 1 ? (IT != NoInterp ? (:(Interpolations.coordlookup(knots[1], I[1])),) : (:(I[1]),)) :
    map(d->Interpolations.iextract(IT, d) != NoInterp ? :(Interpolations.coordlookup(knots[$d], I[$d])) : :(I[$d]), 1:N)

interprange(::Type{IT}, N::Integer) where {IT} = _interprange((), IT, N)
function _interprange(out, IT, N)
    if length(out) < N
        return _interprange((out..., interprange(Interpolations.iextract(IT, length(out)+1))), IT, N)
    end
    out
end
interprange(::Type{NoInterp}) = 0:0
interprange(::Type{BSpline{Constant}}) = 0:0
interprange(::Type{BSpline{Linear}}) = 0:1
interprange(::Type{BSpline{Q}}) where {Q<:Quadratic} = -1:1
interprange(::Type{BSpline{C}}) where {C<:Cubic} = -1:2

coef_gen(::Type{IT}, d::Integer, offsets::CartesianIndex{N}) where {IT,N} =
    coef_gen(Interpolations.iextract(IT, d+1), IT, d+1, offsets)

function coef_gen(::Type{BSpline{D}}, ::Type{IT}, d::Integer, offsets::CartesianIndex{N}) where {D<:Degree,IT<:DimSpec{BSpline}, N}
    if d <= N
        sym = offsets[d] == -1 ? Symbol("cm_",d) :
              offsets[d] ==  0 ? Symbol("c_",d) :
              offsets[d] ==  1 ? Symbol("cp_",d) :
              offsets[d] ==  2 ? Symbol("cpp_",d) : error("offset $(offsets[d]) unknown")
        return :($sym * $(coef_gen(IT, d, offsets)))
    else
        return 1
    end
end

@generated function penalty_hindsight_data(ϕ1::InterpolatingDeformation{T,N},
                                           ϕ2::InterpolatingDeformation{T,N},
                                           fixed::AbstractArray{T1,N},
                                           moving::AbstractInterpolation{T2,N}) where {T,N,T1,T2}
    uindexes = [:((I[$d]-offsets[$d])/steps[$d] + 1) for d = 1:N]
    ϕ1xindexes = [:(I[$d] + u1[$d]) for d = 1:N]
    ϕ2xindexes = [:(I[$d] + u2[$d]) for d = 1:N]
    quote
        knots = ϕ1.knots
        ϕ2.knots == knots || error("knots of ϕ1 and ϕ2 must be the same, got $knots and $(ϕ2.knots), respectively")
        steps = map(step, knots)
        offsets = map(first, knots)
        valid = 0
        mm1 = mm2 = 0.0
        for I in CartesianRange(indices(fixed))
            fval = fixed[I]
            if isfinite(fval)
                u1 = ϕ1.u[$(uindexes...)]
                u2 = ϕ2.u[$(uindexes...)]
                mval1 = moving[$(ϕ1xindexes...)]
                mval2 = moving[$(ϕ2xindexes...)]
                if isfinite(mval1) && isfinite(mval2)
                    valid += 1
                    diff = float64(fval)-float64(mval1)
                    mm1 += diff^2
                    diff = float64(fval)-float64(mval2)
                    mm2 += diff^2
                end
            end
        end
        mm1/valid, mm2/valid
    end
end

function optimize!(ϕ::InterpolatingDeformation, dp::DeformationPenalty, fixed, moving::AbstractExtrapolation; stepsize = 1.0)
    # Optimize the interpolation coefficients, rather than the values
    # of the deformation at the grid points
    ϕtrial = deepcopy(ϕ)
    g = similar(ϕ.u.itp.coefs)
    objective = ϕ->penalty_hindsight(ϕ, dp, fixed, moving)
    objective2 = (ϕ1,ϕ2)->penalty_hindsight(ϕ1, ϕ2, dp, fixed, moving)
    ∇objective!(g, ϕ) = penalty_hindsight!(g, ϕ, dp, fixed, moving)
    pold = p0 = objective(ϕ)
    while true
        ∇objective!(g, ϕ)
        gmax = mapreduce(v->maximum(abs, v), max, g)
        if gmax == 0 || !isfinite(gmax)
            break
        end
        s = eltype(eltype(g))(stepsize/gmax)
        copy!(ϕtrial.u.itp.coefs, ϕ.u.itp.coefs .- s .* g)
        p, pold = objective2(ϕtrial, ϕ)
        if p >= pold
            break
        end
        copy!(ϕ.u.itp.coefs, ϕtrial.u.itp.coefs)
    end
    ϕ, pold, p0
end

function optimize!(ϕ::GridDeformation, dp::DeformationPenalty, fixed, moving::AbstractExtrapolation; stepsize = 1.0)
    optimize!(interpolate!(ϕ), dp, fixed, moving; stepsize=stepsize)
end

end