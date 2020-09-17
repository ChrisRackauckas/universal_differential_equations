cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Revise
using Optim, Zygote
using OrdinaryDiffEq, Flux, DiffEqFlux, LinearAlgebra, Plots
using DiffEqSensitivity, JLD2
const EIGEN_EST = Ref(0.0f0)
const USE_GPU = Ref(false)
USE_GPU[] = false # the network is small enough such that CPU works just great
#USE_GPU[] = true

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

ann = FastChain(FastDense(N-2,N-2,tanh), FastDense(N-2,N-2,tanh), FastDense(N-2,N-2,tanh),
            FastDense(N-2,N-2,tanh), FastDense(N-2,N-2,tanh)) |> _gpu
pp = initial_params(ann)
lyrs = Flux.params(pp)

function dudt_(u,p,t)
    Φ = ann
    return ops.D1*Φ(u, p) + ops.D2*u
end

function predict_adjoint(fullp)
    Array(concrete_solve(prob, 
			  ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]),
			  u0, fullp, saveat = saveat))
end

function loss_adjoint(fullp)
    pre = predict_adjoint(fullp)
    sum(abs2, training_data - pre)
end

function cb(opt_state:: Optim.OptimizationState)
    display(opt_state.value)
    false
end

cb(trace::Optim.OptimizationTrace) = cb(last(trace))

prob = ODEProblem{false}(dudt_,u0,tspan,pp)
nn = 1
print("here")
saveat = t
soldata = soldata'
soldata = soldata[2:end-1, :]
training_data = _cu(soldata)
epochs = Iterators.repeated((), 20)
concrete_solve(prob, ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]), u0, pp)
learning_rate = ADAM(0.01)
res = DiffEqFlux.sciml_train(loss_adjoint, pp, BFGS(initial_stepnorm=0.01), cb=cb)

#Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
#=
learning_rate = ADAM(0.001)
epochs = Iterators.repeated((), 300)
Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
@time loss_adjoint()
=#
