using TestImages
@static((Sys.islinux() || Sys.iswindows()) && using ImageMagick)  # https://github.com/JuliaImages/ImageView.jl/pull/156#issuecomment-418200062
using Interpolations, RegisterMismatch, RegisterPenalty, RegisterDeformation
using RegisterMismatch: mismatch0, ratio
using RegisterHindsight
using ForwardDiff, StaticArrays
using Test

if !isdefined(@__MODULE__, :test_hindsight)
    include("utils.jl")
end

@testset "1-dimensional" begin
    fixed = sin.(range(0, stop=4π, length=40))
    # Create a circumstance where we should be able to get the answer almost exactly
    u = reshape([0.3, -1.2, -0.8], 1, 3)
    ϕref = GridDeformation(u, axes(fixed))
    ϕiref = interpolate!(copy(ϕref))   # *not* the same as `interpolate(ϕref)`
    moving = warp(fixed, ϕiref)
    emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat(OnCell())))), NaN)

    uwrong = [0.0, 0, 0]
    ϕ = similarϕ(ϕiref, uwrong)
    ap = AffinePenalty(ϕ.nodes, 0.0)
    _, p0 = RegisterHindsight.optimize!(ϕ, ap, fixed, emoving; stepsize=0.1)
    p, _  = RegisterHindsight.optimize!(ϕ, ap, fixed, emoving; stepsize=0.01)
    movw = warp(moving, ϕ)
    r0 = ratio(mismatch0(fixed, moving), 0)
    r  = ratio(mismatch0(fixed, movw), 0)
    @test p0 ≈ r0 rtol=0.02
    @test p  ≈ r  rtol=0.2
    @test p < 0.01*p0


    moving, z_def = jitter(fixed, 0.45);
    λ = 1e-3
    gridsize = (length(fixed),)
    nodes = map(d->range(1, stop=size(fixed,d), length=gridsize[d]), (1:ndims(fixed)...,))
    ap = AffinePenalty{Float64,ndims(fixed)}(nodes, λ)
    u0 = zeros(1, gridsize...)
    ϕ0 = GridDeformation(u0, nodes)

    test_hindsight(fixed, moving, ϕ0, ap)

    u0 = rand(1, gridsize...)./10
    ϕ0 = GridDeformation(u0, nodes)
    test_hindsight(fixed, moving, ϕ0, ap)

    emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat(OnCell())))), NaN)
    ϕ = interpolate!(copy(ϕ0))      # *not* the same as `interpolate(ϕref)`
    p, p0 = RegisterHindsight.optimize!(ϕ, ap, fixed, emoving; stepsize=0.1)
    @test ratio(mismatch0(fixed, moving),1) > ratio(mismatch0(fixed, warp(moving, ϕ)), 1)
end

@testset "2-dimensional" begin
    #2-dimensional images
    inds = (200:300, 200:300)
    λ = 1e-3
    img = map(Float64, testimage("cameraman"))
    fixed = img[inds...]
    moving = img[inds[1].-3,inds[2].-2]
    gridsize = (3,3)
    nodes = map(d->range(1, stop=size(fixed,d), length=gridsize[d]), (1:ndims(fixed)...,))
    ap = AffinePenalty{Float64,ndims(fixed)}(nodes, λ)
    u0 = zeros(2, gridsize...)
    ϕ0 = GridDeformation(u0, nodes)
    test_hindsight(fixed, moving, ϕ0, ap)


    emoving = extrapolate(interpolate(moving, BSpline(Quadratic(Flat(OnCell())))), NaN)
    ϕ = interpolate!(copy(ϕ0))        # *not* the same as `interpolate(ϕref)`
    p, p0 = RegisterHindsight.optimize!(ϕ, ap, fixed, emoving; stepsize=0.1)
    @test ratio(mismatch0(fixed, moving),1) > ratio(mismatch0(fixed, warp(moving, ϕ)), 1)
    for i in eachindex(ϕ.u)
        u = ϕ.u[i]
        @test round(Int, u[1]) == 3
        @test round(Int, u[2]) == 2
    end
end
