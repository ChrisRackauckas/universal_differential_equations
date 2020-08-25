cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Single experiment, move to ensemble further on
# Some good parameter values are stored as comments right now
# because this is really good practice

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using JLD2

## Simulation 

println("Generate data")

function lotka(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2]  - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0f0, 3.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=0.1)

# Ideal data
X = Array(solution)

## Generate Basis for Sparse regression


# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[1]

for i ∈ 1:5
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:5
        if i != j
            push!(polys, (u[1]^i) * (u[2]^j))
            push!(polys, u[2]^i * u[1]^i)
        end
    end
end

# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
basis = Basis(h, u)


## LOOP

loop_losses = []
loop_sparsity = []
loop_error = []
failures = 0

for i in 1:75

    global loop_losses, loop_sparsity, loop_error, basis, X, failures

    println("Starting loop $i")
    Xₙ = X + Float32(1e-3) * randn(eltype(X), size(X))

# Define the neueral network which learns L(x, y, y(t-τ))
# Actually, we do not care about overfitting right now, since we want to
# extract the derivative information without numerical differentiation.
    L = FastChain(FastDense(2, 32, tanh), FastDense(32, 32, tanh), FastDense(32, 2))
    p = initial_params(L)

    function dudt_(u, p, t)
        x, y = u
        z = L(u, p)
        [p_[1] * x + z[1],
    -p_[4] * y + z[2]]
    end

    prob_nn = ODEProblem(dudt_, u0, tspan, p)
    sol_nn = solve(prob_nn, Tsit5(), u0=u0, p=p, saveat=solution.t)

    function predict(θ)
        Array(concrete_solve(prob_nn, Vern7(), u0 = u0, p = θ, saveat=solution.t,
                         abstol=1e-6, reltol=1e-6,
                         sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
    end

# No regularisation right now
    function loss(θ)
        pred = predict(θ)
        sum(abs2, Xₙ .- pred), pred 
    end

# Test
    loss(p)

    losses = []

    callback(θ, l, pred) = begin
        push!(losses, l)
        # if length(losses) % 50 == 0
        #    println("Current loss after $(length(losses)) iterations: $(losses[end])")
        # end
        false
    end

    # First train with ADAM for better convergence
    res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters=200)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")
    
    if losses[end] > 1e5 || any(isnan.(res1.minimizer)) || any(abs.(res1.minimizer) .> 1e5)
        failures += 1
        continue
    end

    # Train with BFGS
    res2 =  try
        DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters=10000)
    catch
        @warn "Optimization in loop $i failed"
        failures += 1
        continue
    end

    println("Final training loss after $(length(losses)) iterations: $(losses[end])")
    push!(loop_losses, losses)

# Plot the data and the approximation
    NNsolution = predict(res2.minimizer)

# Ideal derivatives
    DX = Array(solution(solution.t, Val{1}))

    prob_nn2 = ODEProblem(dudt_, u0, tspan, res2.minimizer)
    _sol = solve(prob_nn2, Tsit5())
    DX_ = Array(_sol(solution.t, Val{1}))

# Ideal data
    L̄ = [-p_[2] * (X[1,:] .* X[2,:])';p_[3] * (X[1,:] .* X[2,:])']
# Neural network guess
    L̂ = L(Xₙ, res2.minimizer)

## Sparse Identification 

# Create an optimizer for the SINDy problem
    opt = SR3()
# Create the thresholds which should be used in the search process
    λ = exp10.(-7:0.1:3)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
    g(x) = x[1] < 1 ? Inf : norm(x, 2)

# Test on uode derivative data
    println("Starting sparse regression")
    Ψ = SINDy(Xₙ[:, 2:end], L̂[:, 2:end], basis, λ,  opt, maxiter=10000, normalize=true, denoise=true, g = g) # Succeed


# Extract the parameter
    p̂ = parameters(Ψ)

# The parameters are a bit off, but the equations are recovered
# Start another SINDy run to get closer to the ground truth
# Create function
    unknown_sys = ODESystem(Ψ)
    unknown_eq = ODEFunction(unknown_sys)

# Just the equations
    b = Basis((u, p, t) -> unknown_eq(u, [1.; 1.], t), u)

# Retune for better parameters -> we could also use DiffEqFlux or other parameter estimation tools here.
    Ψf = SINDy(Xₙ[:, 2:end], L̂[:, 2:end], b, opt=STRRidge(0.01), maxiter=100, convergence_error=1e-18) # Succeed
    
    push!(loop_error, get_error(Ψf))
    push!(loop_sparsity, get_sparsity(Ψf))

    println("Statistics for run $i : \n Error $(loop_error[end])\n  Sparsity $(loop_sparsity[end])")

    if i%25 == 0
        println("Saving...")
        @save "loop_runs.jld2" loop_losses loop_error loop_sparsity failures
    end
end


@save "loop_runs.jld2" loop_losses loop_error loop_sparsity failures