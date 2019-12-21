using Revise
using OrdinaryDiffEq, Flux, DiffEqFlux, LinearAlgebra, CuArrays, Plots
using DiffEqSensitivity
const EIGEN_EST = Ref(0.0f0)
const USE_GPU = Ref(true)
USE_GPU[] = false

_gpu(arg) = USE_GPU[] ? gpu(arg) : cpu(arg)
_cu(arg) = USE_GPU[] ? cu(arg) : identity(arg)

function getops(grid, T=Float32)
    N, dz = length(grid), step(grid)
    d = ones(N-2)
    dl = ones(N-3) # super/lower diagonal
    zv = zeros(N-2) # zero diagonal used to extend D* for boundary condtions

    #D1 first order discritization of ∂_z
    D1= diagm(-1 => -dl, 0 => d)
    D1_B = hcat(zv, D1, zv)
    D1_B[1,1] = -1
    D1_B = _cu((1/dz)*D1_B)

    #D2 discritization of ∂_zz
    D2 = diagm(-1 => dl, 0 => -2*d, 1 => dl)
    κ = 0.05
    D2_B = hcat(zv, D2, zv) #add space for the boundary conditions space for "ghost nodes"
    #we only solve for the interior space steps
    D2_B[1,1] = D2_B[end, end] = 1

    D2_B = _cu((κ/(dz^2)).*D2_B) #add the constant κ as the equation requires and finish the discritization

    #Boundary Conditons matrix QQ
    Q = Matrix{Int}(I, N-2, N-2)
    QQ = _cu(vcat(zeros(1,N-2), Q, zeros(1,N-2)))

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
#
N = 32
grid = range(0, 1, length = N)
tspan = (0.0f0, 1.5f0)
u0 = getu0(grid)
ops = getops(grid)
soldata = ground_truth(grid, tspan)

ddd() = Dense(1<<4,1<<4,tanh)
ann = Chain(Dense(30,1<<4,tanh),
            ddd(),# ddd(), ddd(),
            Dense(1<<4,30,tanh)) |> _gpu
pp = param(Flux.data(DiffEqFlux.destructure(ann)))

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
                              sensealg = SensitivityAlg(quad=false, backsolve=false, checkpointing=true),
                              checkpoints = saveat
                              )

function loss_adjoint()
    pre = predict_adjoint()
    sum(abs2, training_data - pre) #super slow dev the package, watch chris's video, inside the layer do something
end

cb = function ()
    display(extrema(prob.p))
    cur_pred = collect(Flux.data(predict_adjoint()))
    n = size(training_data, 1)
    pl = scatter(1:n,training_data[:,10],label="data", legend =:bottomright)
    scatter!(pl,1:n,cur_pred[:,10],label="prediction")
    pl2 = scatter(saveat,training_data[end,:],label="data", legend =:bottomright)
    scatter!(pl2,saveat,cur_pred[end,:],label="prediction")
    display(plot(pl, pl2, size=(600, 300)))
    display(loss_adjoint())
end

prob = ODEProblem{false}(dudt,u0,tspan,pp)
epochs = Iterators.repeated((), 30) #worksish 500
lyrs = Flux.params(pp)
new_tf = 0.00f0
tolerance = 1.0
#solve PDE in smaller time segments to reduce likelihood of divergence
#nn = 20
nn = 1
for i in 1:nn
    #get updated time
    learning_rate = Descent(0.0005)
    global new_tf += 1.5f0/nn
    tspan = (0.0f0, new_tf) #start and end time with better precision
    global saveat = range(tspan..., length = 30) #time range

    #get data of forward pass
    global training_data = _cu(soldata(saveat))

    #solve the backpass
    global prob = ODEProblem{false}(dudt,u0,tspan,pp)
    Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)

    while (loss_adjoint() > tolerance)
        learning_rate = ADAM(0.001)
        Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
    end
    println("finished loop")
end

epochs = Iterators.repeated((), 300) #worksish 500
learning_rate = ADAM(0.001)
Flux.train!(loss_adjoint, lyrs, epochs, learning_rate, cb=cb)
