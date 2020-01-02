cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Revise
using OrdinaryDiffEq, Flux, DiffEqFlux, LinearAlgebra, Plots
using DiffEqSensitivity, JLD2
const EIGEN_EST = Ref(0.0f0)
const USE_GPU = Ref(false)
#USE_GPU[] = false # the network is small enough such that CPU works just great
USE_GPU[] = true

USE_GPU[] && (using CuArrays)

_gpu(arg) = USE_GPU[] ? gpu(arg) : cpu(arg)
_cu(arg) = USE_GPU[] ? cu(arg) : identity(arg)

function getops(grid, T=Float32)
    N, dz = length(grid), step(grid)
    d = ones(N-2)
    dl = ones(N-3) # super/lower diagonal
    zv = zeros(N-2) # zero diagonal used to extend D* for boundary conditions

    # D1 first order discretization of ∂_z
    D1 = diagm(-1 => -dl, 0 => d)
    D1[1, :] .= 0
    D1[end, :] .= 0

    # D2 discretization of ∂_zz
    D2 = diagm(-1 => dl, 0 => -2*d, 1 => dl)
    κ = 0.05
    D2[1, 1] = D2[end, end] = -1
    D2 = (κ/(dz^2)).*D2 #add the constant κ as the equation requires and finish the discretization
    display(D2), display(D1)

    EIGEN_EST[] = maximum(abs, eigvals(D2))
    D1 = _cu(D1)
    D2 = _cu(D2)
    return (D1=T.(D1), D2=T.(D2))
end

file = jldopen("../DataGeneration/rayleigh_taylor_instability_3d_horizontal_averages.jld2")

Is = keys(file["timeseries/t"])

N = file["grid/Nz"]
Lz = file["grid/Lz"]
Nt = length(Is)

t = Float32.(zeros(Nt))
soldata = Float32.(zeros(Nt, N))

for (i, I) in enumerate(Is)
    t[i] = file["timeseries/t/$I"]
    soldata[i, :] .= file["timeseries/b/$I"]
end

grid = range(0, 1, length = N)
tspan = (t[1], t[end])
u0 = _cu(soldata[1,2:end-1])
ops = getops(grid)

ann = Chain(Dense(N-2,N-2,tanh), Dense(N-2,N-2,tanh), Dense(N-2,N-2,tanh),
            Dense(N-2,N-2,tanh), Dense(N-2,N-2,tanh)) |> _gpu
pp = param(Flux.data(DiffEqFlux.destructure(ann)))
lyrs = Flux.params(pp)

function dudt(u::TrackedArray,p,t)
    Φ = DiffEqFlux.restructure(ann, p)
    return ops.D1*Φ(u) + ops.D2*u
end
function dudt(u::AbstractArray,p,t)
    Φ = DiffEqFlux.restructure(ann, p)
    return ops.D1*Tracker.data(Φ(u)) + ops.D2*u
end

predict_adjoint() = diffeq_adjoint(pp,
                              prob,
                              ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]),
                              u0=u0, saveat = saveat,
                              reltol=1e-5, abstol=1e-6,
                              # no back solve
                              sensealg=SensitivityAlg(quad=false, backsolve=false))

function loss_adjoint()
    pre = predict_adjoint()
    sum(abs2, training_data - pre)
end

cb = function ()
    arr = Array(training_data)
    cur_pred = collect(Flux.data(predict_adjoint()))
    n = size(arr, 1)
    pl = scatter(1:n,arr[:,10],label="data", legend =:bottomright, title = "10th time over space")
    scatter!(pl,1:n,cur_pred[:,10],label="prediction")
    pl2 = scatter(saveat,arr[end÷2,:],label="data", legend =:bottomright, title = "middle point over time")
    scatter!(pl2,saveat,cur_pred[end÷2,:],label="prediction")
    #display(plot(pl, pl2, size=(600, 300)))
    display(loss_adjoint())
end

prob = ODEProblem{false}(dudt,u0,tspan,pp)
nn = 1
saveat = t
soldata = soldata'
soldata = soldata[2:end-1, :]
training_data = _cu(soldata)
epochs = Iterators.repeated((), 20)
learning_rate = ADAM(0.01)
Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
#=
learning_rate = ADAM(0.001)
epochs = Iterators.repeated((), 300)
Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
@time loss_adjoint()
=#
