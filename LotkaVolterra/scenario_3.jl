## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
gr()
using JLD2, FileIO
using Statistics
# Set a random seed for reproduceable behaviour
using Random
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

probpde = ODEProblem(rc_ode, rho_ic(Delta), (0.0f0, T), saveat=dt)
solution = solve(probpde, Tsit5());

# Ideal data
X = Array(solution)
t = solution.t

# Add noise in terms of the mean
x̄ = mean(X, dims = 2)
noise_magnitude = Float32(1e-2)
Xₙ = X .+ (noise_magnitude*x̄) .* randn(eltype(X), size(X))

pl_solution_1 = plot(solution, color = :black, ylabel = "rho(x,t)", label = ["True Data" [nothing for i in 1:Nx-1]...])
scatter!(t, transpose(Xₙ), color = :red,label = ["Noisy Data" [nothing for i in 1:Nx-1]...], legend =:bottomright)
# Create dataset plot
pl_contour = contour(t,x,Xₙ, xlabel = "t", ylabel = "x", fill = (true, :thermal))
pl_initial_data = plot(pl_solution_1, pl_contour, layout = (1,2))
savefig(pl_initial_data, "$(svname)full_data_$(noise_magnitude).pdf")

## Neural Network setup
## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Create the network to recover the reaction
rx_nn = FastChain(
    FastDense(1, 5, rbf),
    FastDense(5, 5, rbf),
    FastDense(5, 5, rbf),
    FastDense(5, 1)
)

#initialize D0 close to D/dx^2
D0 = Float32[6.5]
p1s = initial_params(rx_nn)
p2s = zeros(Float32, 4)
p = [p1s;p2s;D0]

function nn_ode(u,p,t)
    # This is the dircetional derivative
    u_cnn_1   = [p[end-4] * u[end] + p[end-3] * u[1] + p[end-2] * u[2]]
    u_cnn     = [p[end-4] * u[i-1] + p[end-3] * u[i] + p[end-2] * u[i+1] for i in 2:size(u, 1)-1]
    u_cnn_end = [p[end-4] * u[end-1] + p[end-3] * u[end] + p[end-2] * u[1]]

    [rx_nn([ui], p[1:length(p1s)])[1] for ui in u] + p[end] * vcat(u_cnn_1, u_cnn, u_cnn_end)
end

prob_nn = ODEProblem(nn_ode, Xₙ[:, 1], (t[1], t[end]), p)


## Necessary functions

function predict_pde(θ, X = Xₙ[:, 1], T = t)
  # No ReverseDiff if using Flux
  Array(solve(prob_nn,Tsit5(),
        u0 = X, p = θ, tspan = (T[1], T[end]), saveat=T,
        sensealg=ForwardDiffSensitivity()
        ))
end

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

## Train the model
# First train with ADAM
res1 = DiffEqFlux.sciml_train(objective_pde, p, ADAM(0.1f0), cb=callback, maxiters = 200)
# Train with BFGS
res2 = DiffEqFlux.sciml_train(objective_pde, res1.minimizer, BFGS(initial_stepnorm=0.1f0), cb=callback, maxiters = 10000)

p_trained = res2.minimizer

## Data evaluation
tsample = t[1]:1.0*mean(diff(t)):t[end]
X̂ = predict_pde(p_trained, Xₙ[:,1], tsample)

# Create estimates
R̂ = similar(X̂)

for i in 1:size(X̂, 1), j in 1:size(X̂, 2)
    R̂[i,j] = rx_nn(X̂[i,j], p_trained[1:length(p1s)])[1]
end
plot(tsample, R̂', legend = nothing)

## Symbolic regression via sparse regression / SINDy using DataDrivenDiffEq

# Reshape the reactions
@variables u[1:1]
b = Basis(monomial_basis(u, 10), u)
# We assume (via the modeling) a common mapping of the states
# via the reaction. In other words a global function. Hence, we can
# use all measurements as one variable
Xs = vcat(eachrow(X̂[1:end, :])...)
Rs = vcat(eachrow(R̂[1:end, :])...)
opt = STRRidge()#SR3(Float32(1e-2), Float32(1e-2))
λs = Float32.(exp10.(-3:0.01:2))
g(x) = x[1] < 1 ? Inf : norm(x,2)
Ψ = SINDy(Xs, Rs,b, λs, opt, denoise = true, normalize = true, g = g, maxiter = 20000)
println(Ψ)
print_equations(Ψ)


## Save the results

save("$(svname)recovery_$(noise_magnitude).jld2",
    "solution", solution, "X", Xₙ, "t" , t, "neural_network" , rx_nn, "initial_parameters", p, "trained_parameters" , p_trained, # Training
    "losses", losses, "result", Ψ,  # Recovery
    ) # Estimation
