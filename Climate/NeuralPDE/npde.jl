cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Revise
using OrdinaryDiffEq, Flux, DiffEqFlux, LinearAlgebra, Plots
using DiffEqSensitivity
const EIGEN_EST = Ref(0.0f0)
const USE_GPU = Ref(false)
USE_GPU[] = false # the network is small enough such that CPU works just great

USE_GPU[] && (using CuArrays)

_gpu(arg) = USE_GPU[] ? gpu(arg) : cpu(arg)
_cu(arg) = USE_GPU[] ? cu(arg) : identity(arg)

function getops(grid, T=Float32)
    N, dz = length(grid), step(grid)
    d = ones(N-2)
    dl = ones(N-3) # super/lower diagonal
    zv = zeros(N-2) # zero diagonal used to extend D* for boundary conditions

    # D1 first order discretization of ∂_z
    D1= diagm(-1 => -dl, 0 => d)
    D1_B = hcat(zv, D1, zv)
    D1_B[1,1] = -1
    D1_B = _cu((1/dz)*D1_B)

    # D2 discretization of ∂_zz
    D2 = diagm(-1 => dl, 0 => -2*d, 1 => dl)
    κ = 0.05
    D2_B = hcat(zv, D2, zv) #add space for the boundary conditions space for "ghost nodes"
    #we only solve for the interior space steps
    D2_B[1,1] = D2_B[end, end] = 1

    D2_B = _cu((κ/(dz^2)).*D2_B) #add the constant κ as the equation requires and finish the discretization

    # Boundary conditions matrix QQ
    Q = Matrix{Int}(I, N-2, N-2)
    QQ = _cu(vcat(zeros(1,N-2), Q, zeros(1,N-2)))
    #display(QQ)
    #display(D1)
    #display(D1_B)

    D1 = D1_B * QQ
    D2 = D2_B * QQ
    EIGEN_EST[] = maximum(abs, eigvals(Matrix(D2)))
    return (D1=T.(D1), D2=T.(D2))
end

function getu0(grid, T=Float32)
    z = grid[2:N-1]
    f0 = z -> T(exp(-200*(z-0.75)^2))
    u0 = f0.(z) |> _gpu
end

function ode_i(u, p, t)
    Φ = u -> cos.(sin.(u.^3) .+ sin.(cos.(u.^2)))
    return p.D1*Φ(u) + p.D2*u
end

function ground_truth(grid, tspan)
    prob = ODEProblem(ode_i, u0, (tspan[1], tspan[2]+0.1), ops)
    sol = solve(prob, ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]), abstol = 1e-9, reltol = 1e-9)
    return sol
end

N = 32
grid = range(0, 1, length = N)
tspan = (0.0f0, 1.5f0)
u0 = getu0(grid)
ops = getops(grid)
soldata = ground_truth(grid, tspan)

ann = Chain(Dense(30,8,tanh), Dense(8,30,tanh)) |> _gpu
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
                              # no back solve
                              sensealg=SensitivityAlg(quad=false, backsolve=false))

function loss_adjoint()
    pre = predict_adjoint()
    sum(abs2, training_data - pre)
end

cb = function ()
    cur_pred = collect(Flux.data(predict_adjoint()))
    n = size(training_data, 1)
    pl = scatter(1:n,training_data[:,10],label="data", legend =:bottomright,title="Spatial Plot at t=$(saveat[10])")
    scatter!(pl,1:n,cur_pred[:,10],label="prediction")
    pl2 = scatter(saveat,training_data[N÷2,:],label="data", legend =:bottomright, title="Timeseries Plot at Middle X")
    scatter!(pl2,saveat,cur_pred[N÷2,:],label="prediction")
    display(plot(pl, pl2, size=(600, 300)))
    display(loss_adjoint())
end

prob = ODEProblem{false}(dudt,u0,tspan,pp)
epochs = Iterators.repeated((), 30)
nn = 1
saveat = range(tspan..., length = 30) #time range
training_data = _cu(soldata(saveat))
epochs = Iterators.repeated((), 200)
learning_rate = ADAM(0.01)
Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
learning_rate = ADAM(0.001)
epochs = Iterators.repeated((), 1000)
Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
cb()

prob2 = ODEProblem{false}(dudt,u0,(0f0,10f0),pp)
@time full_sol = solve(prob2,
                       ROCK2(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]),
                       saveat = saveat, abstol=1e-4, reltol=1e-2)

cur_pred = collect(Flux.data(predict_adjoint()))
n = size(training_data, 1)
pl = scatter(1:n,training_data[:,10],label="data", legend =:bottomright)
scatter!(pl,1:n,cur_pred[:,10],label="prediction",title="Spatial Plot at t=$(saveat[10])")
pl2 = scatter(saveat,training_data[N÷2,:],label="data", legend =:bottomright, title="Time Series Plot: Middle X")
scatter!(pl2,saveat,cur_pred[N÷2,:],label="prediction")
plot(pl, pl2, size=(600, 300))
savefig("npde_fit.pdf")
