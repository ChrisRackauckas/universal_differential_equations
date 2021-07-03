## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, Optim
using DiffEqFlux, Flux
using Plots
gr()
using JLD2, FileIO
using Statistics
# Set a random seed for reproduceable behaviour
using Random
Random.seed!(2345)

#### NOTE
# Since the recent release of DataDrivenDiffEq v0.6.0 where a complete overhaul of the optimizers took
# place, SR3 has been used. Right now, STLSQ performs better and has been changed.


# Create a name for saving ( basically a prefix )
svname = "Scenario_2_"

## Data generation
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0f0,6.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-6, reltol=1e-6, saveat = 0.1)

scatter(solution, alpha = 0.25)
plot!(solution, alpha = 0.5)

# Ideal data
X = Array(solution)
t = solution.t
# Add noise in terms of the mean
x̄ = mean(X, dims = 2)
noise_magnitude = Float32(1e-2)
Xₙ = X .+ (noise_magnitude*x̄) .* randn(eltype(X), size(X))

# Subsample the data in y
# We assume we have only 5 measurements in y, evenly distributed
ty = collect(t[1]:Float32(6/5):t[end])
# Create datasets for the different measurements
round(Int64, mean(diff(ty))/mean(diff(t)))
XS = zeros(eltype(X), length(ty)-1, floor(Int64, mean(diff(ty))/mean(diff(t)))+1) # All x data
TS = zeros(eltype(t), length(ty)-1, floor(Int64, mean(diff(ty))/mean(diff(t)))+1) # Time data
YS = zeros(eltype(X), length(ty)-1, 2) # Just two measurements in y

for i in 1:length(ty)-1
    idxs = ty[i].<= t .<= ty[i+1]
    XS[i, :] = Xₙ[1, idxs]
    TS[i, :] = t[idxs]
    YS[i, :] = [Xₙ[2, t .== ty[i]]'; Xₙ[2, t .== ty[i+1]]]
end

XS

scatter!(t, transpose(Xₙ))
## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Define the network 2->5->5->5->2
U = FastChain(
    FastDense(2,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,2)
)
# Get the initial parameters, first is linear decay
p = [rand(Float32); initial_params(U)]

# Define the hybrid model
function ude_dynamics!(du,u, p, t, p_true)
    û = U(u, p[2:end]) # Network prediction
    du[1] = p_true[1]*u[1] + û[1]
    # We assume a linear decay rate for the predator
    du[2] = -p[1]*u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du,u,p,t) = ude_dynamics!(du,u,p,t,p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!,Xₙ[:, 1], tspan, p)

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
function loss(θ)
    # Start with a regularization on the network
    l = convert(eltype(θ), 1e-3)*sum(abs2, θ[2:end]) ./ length(θ[2:end])
    for i in 1:size(XS,1)
        X̂ = predict(θ, [XS[i,1], YS[i,1]], TS[i, :])
        # Full prediction in x
        l += sum(abs2, XS[i,:] .- X̂[1,:])
        # Add the boundary condition in y
        l += abs(YS[i, 2] .- X̂[2, end])
    end
    return l
end

# Container to track the losses
losses = Float32[]

# Callback to show the loss during training
callback(θ,l) = begin
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

## Training

# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS
res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.1f0), cb=callback, maxiters = 300)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Train with BFGS
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 10000, g_tol = 1e-10)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Plot the losses
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig(pl_losses, joinpath(pwd(), "plots", "$(svname)_losses.pdf"))

# Rename the best candidate
p_trained = res2.minimizer

## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):mean(diff(solution.t))/2:last(solution.t)
X̂ = predict(p_trained, Xₙ[:, 1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), ylabel = "t", xlabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(t, X[1,:], color = :black, label = "Measurements")
ymeasurements = unique!(vcat(YS...))
tmeasurements = unique!(vcat([[ts[1], ts[end]] for ts in eachrow(TS)]...))
scatter!(tmeasurements, ymeasurements, color = :black, label = nothing, legend = :topleft)
savefig(pl_trajectory, joinpath(pwd(), "plots", "$(svname)_trajectory_reconstruction.pdf"))

# Ideal unknown interactions of the predictor
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])';p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
Ŷ = U(X̂,p_trained[2:end])

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing], legend = :topleft)
savefig(pl_reconstruction, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction.pdf"))

# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", color = :red, label = nothing)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))
savefig(pl_missing, joinpath(pwd(), "plots", "$(svname)_missingterm_reconstruction_and_error.pdf"))
pl_overall = plot(pl_trajectory, pl_missing)
savefig(pl_overall, joinpath(pwd(), "plots", "$(svname)_reconstruction.pdf"))
## Symbolic regression via sparse regression ( SINDy based )

# Create a Basis
@variables u[1:2]
# Generate the basis functions, multivariate polynomials up to deg 5
# and sine
b = [polynomial_basis(u, 5); sin.(u)]
basis = Basis(b, u)

# Create the thresholds which should be used in the search process
λ = Float32.(exp10.(-3:0.01:5))
# Create an optimizer for the SINDy problem
opt = STLSQ(λ)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
g(x) = x[1] <= 1 ? Inf : 2*x[1]-2*log(x[2])

# Define different problems for the recovery
full_problem = ContinuousDataDrivenProblem(solution)
ideal_problem = ContinuousDataDrivenProblem(X̂, ts, DX = Ȳ)
nn_problem = ContinuousDataDrivenProblem(X̂, ts, DX = Ŷ)
# Test on ideal derivative data for unknown function ( not available )
println("Sparse regression")
full_res = solve(full_problem, basis, opt, maxiter = 10000, progress = true)
ideal_res = solve(ideal_problem, basis, opt, maxiter = 10000, progress = true)
nn_res = solve(nn_problem, basis, opt, maxiter = 10000, progress = true)
# Store the results
results = [full_res; ideal_res; nn_res]
# Show the results
map(println, results)
# Show the results
map(println ∘ result, results)
# Show the identified parameters
map(println ∘ parameter_map, results)

# Define the recovered, hyrid model
function recovered_dynamics!(du,u, p, t, p_true)
    û = nn_res(u, p) # Network prediction
    du[1] = p_true[1]*u[1] + û[1]
    du[2] = -p_trained[1]*u[2] + û[2] # Learned paramter + prediction
end

# Closure with the known parameter
estimated_dynamics!(du,u,p,t) = recovered_dynamics!(du,u,p,t,p_)

estimation_prob = ODEProblem(estimated_dynamics!, u0, tspan, parameters(nn_res))
estimate = solve(estimation_prob, Tsit5(), saveat = t)

# Plot
plot(solution)
plot!(estimate)

## Simulation

# Look at long term prediction
t_long = (0.0f0, 50.0f0)
estimation_prob = ODEProblem(estimated_dynamics!, u0, t_long, parameters(nn_res))
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
plot(estimate_long)

true_prob = ODEProblem(lotka!, u0, t_long, p_)
true_solution_long = solve(true_prob, Tsit5(), saveat = estimate_long.t)
plot!(true_solution_long)

## Save the results
save(joinpath(pwd(), "results" ,"$(svname)recovery_$(noise_magnitude).jld2"),
    "solution", solution, "X", Xₙ, "t" , t, "neural_network" , U, "initial_parameters", p, "trained_parameters" , p_trained, # Training
    "losses", losses, "result", nn_res, "recovered_parameters", parameters(nn_res), # Recovery
    "long_solution", true_solution_long, "long_estimate", estimate_long) # Estimation


## Post Processing and Plots

c1 = 3 # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple # RGBA(153/255,50/255,204/255,1) # Purple

p1 = plot(t,abs.(Array(solution) .- estimate)' .+ eps(Float32),
          lw = 3, yaxis = :log, title = "Timeseries of UODE Error",
          color = [3 :orange], xlabel = "t",
          label = ["x(t)" "y(t)"],
          titlefont = "Helvetica", legendfont = "Helvetica",
          legend = :topright)

# Plot L₂
p2 = plot(X[1,:], X[2,:], Ŷ[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = c1,
     label = "Neural Network", xaxis = "x", yaxis="y",
     titlefont = "Helvetica", legendfont = "Helvetica",
     legend = :bottomright)
plot!(X[1,:], X[2,:], Ȳ[2,:], lw = 3, label = "True Missing Term", color=c2)

p3 = scatter(t, Xₙ[1,:], color = [c1], label = "x data",
             title = "Extrapolated Fit From Incomplete Training Data",
             titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

scatter!(ty, [YS[:, 1]; YS[end, end]], color = c2, label = "y data", markersize = 5)
plot!(p3,true_solution_long, color = [c1 c2], linestyle = :dot, lw=5, label = ["True x(t)" "True y(t)"])
plot!(p3,estimate_long, color = [c3 c4], lw=2, label = ["Estimated x(t)" "Estimated y(t)"])
plot!(p3,[5.99,6.01],[0.0,maximum(hcat(Array(solution),Array(estimate_long)))],lw=1,color=:black, label = nothing)
annotate!([(3,12,text("Training \nData", 10, :center, :top, :black, "Helvetica"))])
l = @layout [grid(1,2)
             grid(1,1)]
plot(p1,p2,p3,layout = l)

savefig(joinpath(pwd(),"plots","$(svname)full_plot.pdf"))
