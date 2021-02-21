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
using DelimitedFiles
# Set a random seed for reproduceable behaviour
using Random
Random.seed!(5443)

## Data Preprocessing
# The data has been taken from https://jmahaffy.sdsu.edu/courses/f00/math122/labs/labj/q3v1.htm
# Originally published in
hudson_bay_data = readdlm("hudson_bay_data.dat", '\t', Float32, '\n')
# Measurements of prey and predator
Xₙ = Matrix(transpose(hudson_bay_data[:, 2:3]))
t = hudson_bay_data[:, 1] .- hudson_bay_data[1, 1]
# Normalize the data; since the data domain is strictly positive
# we just need to divide by the maximum
xscale = maximum(Xₙ, dims =2)
Xₙ .= 1f0 ./ xscale .* Xₙ
# Time from 0 -> n
tspan = (t[1], t[end])

# Plot the data
scatter(t, transpose(Xₙ), xlabel = "t [a]", ylabel = "x(t), y(t)")
plot!(t, transpose(Xₙ), xlabel = "t", ylabel = "x(t), y(t)")

## Direct Identification via SINDy + Collocation
# Create a collocation
dx̂,x̂ = collocate_data(Xₙ,t, GaussianKernel())
# Look at the collocation
plot(t, dx̂')
# Perform sindy

# Create a Basis
@variables u[1:2]

# Generate the basis functions, multivariate polynomials up to deg 5
# and sine
b = [polynomial_basis(u, 5); sin.(u)]
basis = Basis(b, u)
# Create an optimizer for the SINDy problem
opt = SR3(Float32(1e-2), Float32(1e-2))
# Create the thresholds which should be used in the search process
λ = Float32.(exp10.(-7:0.1:3))
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
g(x) = x[1] < 1 ? Inf : norm(x, 2)
# Test on derivative data
Ψ = SINDy(x̂, dx̂, basis, λ,  opt, g = g, maxiter = 50000, normalize = true, denoise = true) # Succeed
println(Ψ)
print_equations(Ψ) # Fails
b2 = Basis((u,p,t)->Ψ(u,ones(length(parameters(Ψ))),t),u, linear_independent = true)
Ψ = SINDy(x̂, dx̂, b2, λ,  opt, g = g, maxiter = 50000, normalize = true, denoise = true) # Succeed
println(Ψ)
print_equations(Ψ) # Fails
parameters(Ψ)
## UDE Approach
# Subsample the data in y -> initial fitting strategy (batching)
# We assume we have only 5 measurements in y, evenly distributed
ty = collect(t[1]:Float32(t[end]/5):t[end])
# Create datasets for the different measurements
t
XS = zeros(Float32, length(ty)-1, floor(Int64, mean(diff(ty))/mean(diff(t)))+1) # All x data
TS = zeros(Float32, length(ty)-1, floor(Int64, mean(diff(ty))/mean(diff(t)))+1) # Time data
YS = zeros(Float32, length(ty)-1, 2) # Just two measurements in y

for i in 1:length(ty)-1
    idxs = ty[i].<= t .<= ty[i+1]
    XS[i, :] = Xₙ[1, idxs]
    TS[i, :] = t[idxs]
    YS[i, :] = [Xₙ[2, t .== ty[i]]'; Xₙ[2, t .== ty[i+1]]]
end

## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Define the network 2->5->5->5->2
U = FastChain(
    FastDense(2,5,rbf), FastDense(5,5, rbf), FastDense(5,5, tanh), FastDense(5,2)
)

# Get the initial parameters, first two is linear birth / decay of prey and predator
p = [rand(Float32,2); initial_params(U)]

# Define the hybrid model
function ude_dynamics!(du,u, p, t)
    û = U(u, p[3:end]) # Network prediction
    # We assume a linear birth rate for the prey
    du[1] = p[1]u[1] + û[1]
    # We assume a linear decay rate for the predator
    du[2] = -p[2]*u[2] + û[2]
end

# Define the problem
prob_nn = ODEProblem(ude_dynamics!,Xₙ[:, 1], tspan, p)

## Function to train the network
# Define a predictor
function predict(θ, X = Xₙ[:,1], T = t)
    Array(solve(prob_nn, Vern7(), u0 = X, p=θ,
                tspan = (T[1], T[end]), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

# Multiple shooting like loss
function shooting_loss(θ)
    # Start with a regularization on the network
    l = convert(eltype(θ), 1e-3)*sum(abs2, θ[3:end]) ./ length(θ[3:end])
    for i in 1:size(XS,1)
        X̂ = predict(θ, [XS[i,1], YS[i,1]], TS[i, :])
        # Full prediction in x
        l += sum(abs2, XS[i,:] .- X̂[1,:])
        # Add the boundary condition in y
        l += abs2(YS[i, 2] .- X̂[2, end])
    end

    return l
end

function loss(θ)
    X̂ = predict(θ)
    sum(abs2, Xₙ - X̂) + convert(eltype(θ), 1e-3)*sum(abs2, θ[3:end]) ./ length(θ[3:end])
end

# Container to track the losses
losses = Float32[]

# Callback to show the loss during training
callback(θ,l) = begin
    push!(losses, l)
    if length(losses)%5==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

## Training -> First shooting / batching to get a rough estimate

# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS
res1 = DiffEqFlux.sciml_train(shooting_loss, p, ADAM(0.1f0), cb=callback, maxiters = 100)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Train with BFGS to achieve partial fit of the data
res2 = DiffEqFlux.sciml_train(shooting_loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 200)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Full L2-Loss for full prediction
res3 = DiffEqFlux.sciml_train(loss, res2.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 10000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res3.minimizer

## Analysis of the trained network
# Interpolate the solution
tsample = t[1]:0.5:t[end]
X̂ = predict(p_trained, Xₙ[:,1], tsample)
# Trained on noisy data vs real solution
plot(t, transpose(Xₙ), color = :black, label = ["Measurements" nothing], xlabel = "t", ylabel = "x(t), y(t)")
plot!(tsample, transpose(X̂), color = :red, label = ["Interpolation" nothing])

# Neural network guess
Ŷ = U(X̂,p_trained[3:end])

scatter(tsample, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])

## Symbolic regression via sparse regression ( SINDy based )

# Create a Basis
@variables u[1:2]

# Generate the basis functions, multivariate polynomials up to deg 5
# and sine
b = [polynomial_basis(u, 5); sin.(u)]
basis = Basis(b, u)

# Create an optimizer for the SINDy problem
opt = STRRidge()
# Create the thresholds which should be used in the search process
λ = Float32.(exp10.(-7:0.1:3))
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
g(x) = x[1] < 1 ? Inf : norm(x, 2)

# Test on uode derivative data
println("SINDy on learned, partial, available data")
Ψ = SINDy(X̂, Ŷ, basis, λ,  opt, g = g, maxiter = 50000, normalize = true, denoise = true)
println(Ψ)
print_equations(Ψ)
# Extract the parameter
p̂ = parameters(Ψ)
println("First parameter guess : $(p̂)")


# Define the recovered, hyrid model with the rescaled dynamics
function recovered_dynamics!(du,u, p, t)
    û = Ψ(u, p[3:4]) # Network prediction
    du[1] = p[1]*u[1] + û[1]
    du[2] = -p[2]*u[2] + û[2]
end

p_model = [p_trained[1:2];p̂]
estimation_prob = ODEProblem(recovered_dynamics!, Xₙ[:, 1], tspan, p_model)
estimate = solve(estimation_prob, Tsit5(), saveat = t)

# Plot
plot(t, transpose(Xₙ), color = :black, label = ["Measurements" nothing], xlabel = "t", ylabel = "x(t), y(t)")
scatter!(estimate, color = :red, label = ["Recovered" nothing])


## Simulation

# Look at long term prediction
t_long = (0.0f0, 50.0f0)
estimation_prob = ODEProblem(recovered_dynamics!, Xₙ[:, 1], t_long, p_model)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.25)
plot(estimate_long)

## Save the results
save(joinpath(pwd(),"results","Hudson_Bay_recovery.jld2"),
    "X", Xₙ, "t" , t, "neural_network" , U, "initial_parameters", p, "trained_parameters" , p_trained, # Training
    "losses", losses, "result", Ψ, "recovered_parameters", p̂, # Recovery
    "model", recovered_dynamics!, "model_parameter", p_model,
    "long_estimate", estimate_long) # Estimation
