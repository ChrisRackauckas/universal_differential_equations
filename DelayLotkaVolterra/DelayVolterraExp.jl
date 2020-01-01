cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Single experiment, move to ensemble further on
# Some good parameter values are stored as comments right now
# because this is really good practice

using DifferentialEquations
using Flux, Tracker
using LinearAlgebra
using DiffEqFlux
using Plots
gr()



function lotka_delay(du, u, h, p, t)
    α, β, γ, δ = p
    du[1] = α*h(p, t-1.0)[1] - β*u[1]*u[2]
    du[2] = γ*u[1]*h(p, t-0.5)[2] - δ*u[2]
end

# Define the experimental parameter
# Empirical upper bound is 7
tspan = (0.0f0,7.0f0)
# Good initial value ( trained save to Loss ≈ 1 ) has been [0.8, 0.3]
u0 = Float32[4.0; 0.3]
#p = Float32[0.5, 0.5, 0.7, 0.3]
p = Float32[0.5, 0.5, 0.1, 0.1]
h(p, t) = u0

prob = DDEProblem(lotka_delay, u0, h, tspan, p, constant_lags = Float32[0.5, 1.0])
solution = solve(prob, Tsit5(), saveat = 0.1f0)
plot(solution)

# Initial condition and parameter for the Neural DDE
u0_ = Tracker.param(u0)
p_ = param(p)

# Assume steady state
g(p, t) = u0_

# Define the neueral network which learns L(x, y, y(t-τ))
# Actually, we do not care about overfitting right now, since we want to
# extract the derivative information without numerical differentiation.
ann = Chain(Dense(3,20,swish), Dense(20, 20, swish),Dense(20, 20, swish), Dense(20, 2))

function dudt_(u::TrackedArray, h ,p,t)
    x, y = u

    # I drop the history here and just train the nn to match the derivative
    #z = Flux.data.([u..., h(p, t-0.5f0)[2]])
    # Seems to work
    z = [u..., h(p, t-0.5f0)[2]] |> Tracker.collect

    Tracker.collect([p[1]*h(p, t-1.0f0)[1] + ann(z)[1],
        -p[4]*y + ann(z)[2]])
end

function dudt_(u::AbstractArray, h ,p,t)
    x, y = u
    z = Flux.data.([u..., h(p, t-0.5f0)[2]])
    Tracker.data.([p[1]*h(p, t-1.0f0)[1] + Flux.data.(ann(z)[1]),
        -p[4]*y + Flux.data.(ann(z)[2])])
end

prob = DDEProblem(dudt_,u0_, g, tspan, p_, constant_lags = Float32[0.5, 1.0])
s = diffeq_rd(p, prob, Rosenbrock23())
plot(Flux.data.(s)')

function predict_rd()
    diffeq_rd(p_, prob, Rosenbrock23(), saveat = solution.t)
end

function predict_rd(sol)
    diffeq_rd(p_, prob, u0 = param(sol[:,1]), Rosenbrock23(), saveat = sol.t)
end

# No regularisation right now
loss_rd() = sum(abs2, solution[:,:] .- predict_rd()[:,:]) #+ 1e-5*sum(sum.(abs, params(ann)))
loss_rd()

# AdamW forgets, which might be nice since the nn changes topology over time
opt = ADAM(1e-2)

# Just use this right now, switch to tensorboard later
const losses = []

callback() = begin
    push!(losses, Flux.data(loss_rd()))
    if length(losses) % 50 == 0
        display(losses[end])
    end
end

# Train the neural DDE
for i in 1:1000 -length(losses)
    Flux.train!(loss_rd, params(ann), [()], opt, cb = callback)
end

# Show the training evaluation
plot(losses, yaxis = :log, ylabel = "Loss", xlabel = "Episodes")

# Plot the data and the approximation
plot(solution.t, Flux.data.(predict_rd()'))
plot!(solution)

# Plot the error
plot(abs.(solution[:,:] .- Flux.data.(predict_rd()[:,:]))' .+ eps(Float32), yaxis = :log)


# TODO Double check this, should be right way to move to
# Delay Coordinat manifold
Z = cat(solution[:,solution.t .> 0.5] ,solution[:, solution.t .< tspan[2]-0.5], dims = 1)[[1,2,4],:]
L = Flux.data(ann(Z))
# Get the analytical solution
l1 = -p[2]*Z[1,:].*Z[2,:]
l2 = p[3]*Z[1,:].*Z[3,:]
# Plot L₁
plot3d(Z[1,:], Z[2,:], L[1,:], xlabel = "x", ylabel = "y", zlabel = "L₁")
plot3d!(Z[1,:], Z[2,:], l1)
# Plot L₂
plot3d(Z[1,:], Z[2,:],  L[2,:])
plot3d!(Z[1,:], Z[2,:], l2)
