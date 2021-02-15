cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux

using JLD2, FileIO
using Statistics
# Set a random seed for reproduceable behaviour
using Random
Random.seed!(1234)

## Data generation
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

@info "Generate data "

# Define the experimental parameter
tspan = (0.0f0,3.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

# Ideal data
X = Array(solution)
t = solution.t
Xₙ = similar(X)
# Add noise in terms of the mean
x̄ = mean(X, dims = 2)

# Create a function to adapt the noise magnitude
function noisy_magnitude(iteration)
    iteration <= 100 && return Float32(1e-3)
    iteration <= 200 && return Float32(5e-3)
    iteration <= 300 && return Float32(1e-2)
    iteration <= 400 && return Float32(2.5e-2)
    return Float32(5e-2)
end

## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

include("./loop_recoveries.jl")

@info "Start evaluation loops"
for i in 1:1:5
    Xₙ .= X .+ (noisy_magnitude(i)*x̄) .* randn(eltype(X), size(X)...)
    recover_dynamics(X1n, t1, "Scenario_1", "$i")
end
@info "Finished evaluation loops"
