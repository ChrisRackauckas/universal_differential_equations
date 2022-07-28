## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL #OptimizationFlux for ADAM and OptimizationOptimJL for BFGS
using DiffEqSensitivity
using Lux
using Plots
gr()
using JLD2, FileIO
using Statistics
# Set a random seed for reproduceable behaviour
using Random
rng = Random.default_rng()
Random.seed!(1235)

# Create a name for saving ( basically a prefix )
svname = "Scenario_3_"

## Data Generation
# Parameters
D = 0.01f0; #diffusion
r = 1.0f0; #reaction rate

# Domain
X = 1.0f0; T = 5f0;
dx = 0.04f0; dt = T/10;
x = collect(0:dx:X);
t = collect(0:dt:T);
Nx = Int64(X/dx+1);
Nt = Int64(T/dt+1);

#initial conditions
Amp = 1.0f0;
Delta = 0.5f0;
rho_ic(Δ) = Amp*(tanh.((x .- (0.5f0 - Δ/2))/(Δ/10)) - tanh.((x .- (0.5f0 + Δ/2))/(Δ/10)))/2

# Reaction term
reaction(u) = r * u .* (1f0 .- u)
# Difference Op
lap = Float32.(diagm(0 => -2.0f0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2)
#Periodic BC
lap[1,end] = 1.0f0/dx^2
lap[end,1] = 1.0f0/dx^2

function rc_ode(rho, p, t)
    #finite difference
    D * lap * rho + reaction.(rho)
end

# Generate random measurements
probpde = ODEProblem(rc_ode, rho_ic(Delta), (0.0f0, T), saveat=dt)
solution = solve(probpde, Tsit5());

# Ideal data
X = Array(solution)
t = solution.t

# Add noise in terms of the mean
x̄ = mean(X, dims = 2)
noise_magnitude = Float32(5e-3)
Xₙ = X .+ (noise_magnitude*x̄) .* randn(eltype(X), size(X))



pl_solution_1 = plot(solution, color = :black, ylabel = "rho(x,t)", label = ["True Data" [nothing for i in 1:Nx-1]...])
scatter!(t, transpose(Xₙ), color = :red,label = ["Noisy Data" [nothing for i in 1:Nx-1]...], legend =:bottomright)
# Create dataset plot
pl_contour = contour(t,x,Xₙ, xlabel = "t", ylabel = "x", fill = (true, :thermal))
pl_initial_data = plot(pl_solution_1, pl_contour, layout = (1,2))
savefig(pl_initial_data, joinpath(pwd(), "plots","$(svname)full_data_$(noise_magnitude).pdf"))

## Neural Network setup
## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Create the network to recover the reaction
rx_nn = Lux.Chain(
    Lux.Dense(1, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 1)
)

#initialize D0 close to D/dx^2
D0 = Float32[6.5]

# Get the initial parameters and state variables of the model
p_nn, st_nn = Lux.setup(rng, rx_nn)
p2s = zeros(Float32, 4)
p = ComponentVector(;
    ude = p_nn,
    p2s = p2s,
    D0 = D0
)


function nn_ode(u,p,t)
    # This is the dircetional derivative

    ps_ = p.p2s
    u_cnn_1   = [ps_[end-3] * u[end] + ps_[end-2] * u[1] + ps_[end-1] * u[2]]
    u_cnn     = [ps_[end-3] * u[i-1] + ps_[end-2] * u[i] + ps_[end-1] * u[i+1] for i in 2:size(u, 1)-1]
    u_cnn_end = [ps_[end-3] * u[end-1] + ps_[end-2] * u[end] + ps_[end-1] * u[1]]

    reduce(vcat, map(u) do ui 
        rx_nn([ui], p.ude, st_nn)[1]
    end) + p.D0 .* vcat(u_cnn_1, u_cnn, u_cnn_end)
end

prob_nn = ODEProblem(nn_ode, Xₙ[:, 1], (t[1], t[end]), p)
nn_ode(rho_ic(Delta), p, 0.0)

## Necessary functions

function predict_pde(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                sensealg = ForwardDiffSensitivity(convert_tspan = false)
                ))
end

predict_pde(p)

#match data and force the weights of the CNN to add up to zero
function objective_pde(p)
    pred = predict_pde(p)
    sum(abs2, pred .- Xₙ) + abs(sum(p[end-4 : end-2]))
end

# Container to track the loss
losses = Float32[]

# Callback to see the training
callback(θ,l) = begin
    push!(losses, l)
    if length(losses)%1==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->objective_pde(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
res1 = Optimization.solve(optprob, ADAM(0.1), callback=callback, maxiters = 10)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Train with BFGS
optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.001), callback=callback, maxiters = 100)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

p_trained = res2.minimizer

## Data evaluation -> we subsample here!
X̂ = predict_pde(p_trained)
# Trained on noisy data vs real solution
pl_trajectory = plot(t, transpose(X̂), xlabel = "t", ylabel ="rho(x,t)", color = :red,
 label = ["UDE Approximation" [nothing for i in 1:25]...])
scatter!(t, transpose(Xₙ), color = :black, label = ["Measurements"  [nothing for i in 1:25]...], legend = :bottomright)
savefig(pl_trajectory, joinpath(pwd(), "plots", "$(svname)_trajectory_reconstruction.pdf"))


# Create estimates
R̂ = similar(X̂)
R̄ = similar(X̂)
for i in 1:size(X̂, 1), j in 1:size(X̂, 2)
    R̂[i,j] = rx_nn([X̂[i,j]], p_trained.ude, st_nn)[1][1]
    R̄[i,j] = reaction(X̂[i,j])
end

pl_reconstruction = plot(tsample, R̂', label = ["UDE Approximation" [nothing for i in 1:25]...], color = :red, xlabel = "t", ylabel = "Reaction(x,t)")
plot!(tsample, R̄', label = ["True Reaction" [nothing for i in 1:25]...], color = :black)
pl_reconstruction_error = plot(tsample, norm.(eachcol(R̂ - R̄)), color = :red, xlabel = "t", ylabel = "L2 Error")
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))
savefig(pl_missing, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction_and_error.pdf"))
pl_overall = plot(pl_trajectory, pl_missing)
savefig(pl_overall, joinpath(pwd(), "plots", "$(svname)_reconstruction.pdf"))
## Symbolic regression via sparse regression / SINDy using DataDrivenDiffEq

# Reshape the reactions
@variables u[1:1]
b = Basis(monomial_basis(u, 10), u)
# We assume (via the modeling) a common mapping of the states
# via the reaction. In other words a global function. Hence, we can
# use all measurements as one variable
Xs = Matrix(vcat(eachrow(X̂[1:end, :])...)')
Rs = Matrix(vcat(eachrow(R̂[1:end, :])...)')

# Technically this is cheating until a general DataDrivenProblem is defined properly.
# we work with the continuous form
nn_prob = DirectDataDrivenProblem(Xs, Rs)

λs = Float32.(exp10.(-3:0.01:5))
opt = STLSQ(λs)

nn_res = solve(nn_prob, b, opt, maxiters = 1000, progress = true, denoise = true, normalize = false)
# Print the output
println(nn_res)
println(result(nn_res))
println(parameters(nn_res))
## Save the results

save(joinpath(pwd(), "results" ,"$(svname)recovery_$(noise_magnitude).jld2"),
    "solution", solution, "X", Xₙ, "t" , t, "neural_network" , rx_nn, "initial_parameters", p, "trained_parameters" , p_trained, # Training
    "losses", losses, "result", nn_res,  # Recovery
    ) # Estimation
