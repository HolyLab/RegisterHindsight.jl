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

#1-dimensional images
fixed = sin.(range(0, stop=4π, length=40))
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
ϕ = interpolate!(copy(ϕ0))
p, p0 = Main.RegisterHindsight.optimize!(ϕ, ap, fixed, emoving; stepsize=0.1)
@test ratio(mismatch0(fixed, moving),1) > ratio(mismatch0(fixed, warp(moving, ϕ)), 1)

#2-dimensional images
inds = (200:300, 200:300)
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
ϕ = interpolate!(copy(ϕ0))
p, p0 = Main.RegisterHindsight.optimize!(ϕ, ap, fixed, emoving; stepsize=0.1)
@test ratio(mismatch0(fixed, moving),1) > ratio(mismatch0(fixed, warp(moving, ϕ)), 1)
for i in eachindex(ϕ.u)
    u = ϕ.u[i]
    @test round(Int, u[1]) == 3
    @test round(Int, u[2]) == 2
end
