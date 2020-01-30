# add jitter in sampling location, simulating inconsistencies in piezo position when using
# OCPI under certain conditions

RegisterDeformation.floattype(::Type{ForwardDiff.Dual{Tag,V,N}}) where {Tag,V,N} =
    RegisterDeformation.floattype(V)

function jitter(img::AbstractVector, npix::Real)
    etp = extrapolate(interpolate(img, BSpline(Linear())), Flat())
    out = zeros(eltype(img), size(img))
    z_def = Float64[]
    r = 0.0
    for i in eachindex(img)
        # To ensure that our sampling approximately satisfies the Nyquist
        # criterion, smooth r
        Δz = (2*rand()-1)*npix
        r = (3r + Δz)/4  # exponential filter with τ ≈ 4
        push!(z_def, r)
        out[i] = etp(i+r)
    end
    return out, z_def
end

function test_hindsight(fixed, moving, ϕ0, ap)
    ϕ = interpolate!(copy(ϕ0))   # not same as interpolate(ϕ)
    coefs = RegisterDeformation.getcoefs(ϕ)
    udata = RegisterDeformation.convert_from_fixed(coefs)
    g_data = similar(coefs)

    nanval = RegisterDeformation.nanvalue(eltype(moving))
    emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat(OnCell())))), nanval)

    # Setup for ForwardDiff, which only works on vectors of parameters
    function penaltydata(newcoefs)
        ϕnew = similarϕ(ϕ, newcoefs)
        return RegisterHindsight.penalty_hindsight_data(ϕnew, fixed, emoving)
    end
    function penaltyreg(newcoefs)
        ϕnew = similarϕ(ϕ, newcoefs)
        return RegisterHindsight.penalty_hindsight_reg(ap, ϕ)
    end

    # Check penalty value consistency for low-level calls
    pdata1 = RegisterHindsight.penalty_hindsight_data(ϕ, fixed, emoving)
    preg1 = RegisterHindsight.penalty_hindsight_reg(ap, ϕ)
    ptotal = RegisterHindsight.penalty_hindsight(ϕ, ap, fixed, emoving)
    @test ptotal == pdata1 + preg1
    pdata2 = RegisterHindsight.penalty_hindsight_data!(g_data, ϕ, fixed, emoving) #fully optimized version
    @test pdata1 == pdata2

    g = ForwardDiff.gradient(penaltydata, udata)
    @test g_data ≈ RegisterDeformation.convert_to_fixed(eltype(g_data), g)

    #test that gradients sum properly
    g_total, g_reg = similar(g_data), similar(g_data)
    RegisterHindsight.penalty_hindsight!(g_total, ϕ, ap, fixed, emoving)
    RegisterHindsight.penalty_hindsight_reg!(g_reg, ap, ϕ)
    @test g_total == g_data .+ g_reg
end
