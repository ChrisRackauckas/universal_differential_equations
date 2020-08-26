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
using Plots
gr()

function lotka(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0f0,3.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

scatter(solution, alpha = 0.25)
plot!(solution, alpha = 0.5)

# Ideal data
X = Array(solution)
# Add noise to the data
println("Generate noisy data")
Xₙ = X + Float32(1e-3)*randn(eltype(X), size(X))

# Define the neueral network which learns L(x, y, y(t-τ))
# Actually, we do not care about overfitting right now, since we want to
# extract the derivative information without numerical differentiation.
L = FastChain(FastDense(2, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 2))
p = initial_params(L)

function dudt_(u, p,t)
    x, y = u
    z = L(u,p)
    [p_[1]*x + z[1],
    -p_[4]*y + z[2]]
end

prob_nn = ODEProblem(dudt_,u0, tspan, p)
sol_nn = solve(prob_nn, Tsit5(), u0 = u0, p = p, saveat = solution.t)

plot(solution)
plot!(sol_nn)

function predict(θ)
    Array(solve(prob_nn, Vern7(), u0 = u0, p=θ, saveat = solution.t,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, Xₙ .- pred), pred 
end

# Test
loss(p)

const losses = []

callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

# First train with ADAM for better convergence
res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 200)

# Train with BFGS
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)

println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Plot the losses
plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")

# Plot the data and the approximation
NNsolution = predict(res2.minimizer)
# Trained on noisy data vs real solution
plot(solution.t, NNsolution')
plot!(solution.t, X')

# Ideal derivatives
DX = Array(solution(solution.t, Val{1}))

prob_nn2 = ODEProblem(dudt_,u0, tspan, res2.minimizer)
_sol = solve(prob_nn2, Tsit5())
DX_ = Array(_sol(solution.t, Val{1}))

# The learned derivatives
plot(DX')
plot!(DX_')

# Ideal data
L̄ = [-p_[2]*(X[1,:].*X[2,:])';p_[3]*(X[1,:].*X[2,:])']
# Neural network guess
L̂ = L(Xₙ,res2.minimizer)

scatter(L̄')
plot!(L̂')

# Plot the error
scatter(abs.(L̄-L̂)', yaxis = :log)

## Sparse Identification 

# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[1]

for i ∈ 1:5
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:5
        if i != j
            push!(polys, (u[1]^i)*(u[2]^j))
            push!(polys, u[2]^i*u[1]^i)
        end
    end
end

# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
basis = Basis(h, u)

# Create an optimizer for the SINDy problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = exp10.(-7:0.1:3)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
g(x) = x[1] < 1 ? Inf : norm(x, 2)
# Test on original data and without further knowledge
println("SINDy on full ideal, unavailable data")
Ψ = SINDy(Xₙ[:, :], DX[:, :], basis, λ, opt, g = g, maxiter = 10000) # Fail
println(Ψ)
print_equations(Ψ)

# Test on ideal derivative data for unknown function ( not available )
println("SINDy on partial ideal, unavailable data")
Ψ = SINDy(Xₙ[:, 1:end], L̄[:, 1:end], basis, λ,opt, g = g, maxiter = 10000) # Succeed
println(Ψ)
print_equations(Ψ)

# Test on uode derivative data
println("SINDy on learned, partial, available data")
Ψ = SINDy(Xₙ[:, 2:end], L̂[:, 2:end], basis, λ,  opt, g = g, maxiter = 10000, normalize = true, denoise = true) # Succeed
println(Ψ)
print_equations(Ψ)

# Extract the parameter
p̂ = parameters(Ψ)
println("First parameter guess : $(p̂)")

# The parameters are a bit off, but the equations are recovered
# Start another SINDy run to get closer to the ground truth
# Create function
unknown_sys = ODESystem(Ψ)
unknown_eq = ODEFunction(unknown_sys)

# Just the equations
b = Basis((u, p, t)->unknown_eq(u, [1.; 1.], t), u)

# Retune for better parameters -> we could also use DiffEqFlux or other parameter estimation tools here.
Ψf = SINDy(Xₙ[:, 2:end], L̂[:, 2:end], b, STRRidge(0.01), maxiter = 100, convergence_error = 1e-18) # Succeed
println(Ψf)
p̂ = parameters(Ψf)
println("Second parameter guess : $(p̂)")

# Create function
recovered_sys = ODESystem(Ψf)
recovered_eq = ODEFunction(recovered_sys)

# Build a ODE for the estimated system
function dudt(du, u, p, t)
    # Add SINDy Term
    α, δ, β, γ = p
    z = recovered_eq(u, [β; γ], t)
    du[1] = α*u[1] + z[1]
    du[2] = -δ*u[2] + z[2]
end

# Create the approximated problem and solution
ps = [p_[[1,4]]; p̂]
approximate_prob = ODEProblem(dudt, u0, tspan, ps)
approximate_solution = solve(approximate_prob, Tsit5(), saveat = 0.01)

# Plot
plot(solution)
plot!(approximate_solution)

## Simulation

# Look at long term prediction
t_long = (0.0, 50.0)
approximate_prob = ODEProblem(dudt, u0, t_long, ps)
approximate_solution_long = solve(approximate_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
plot(approximate_solution_long)

true_prob = ODEProblem(lotka, u0, t_long, p_)
true_solution_long = solve(true_prob, Tsit5(), saveat = approximate_solution_long.t)
plot!(true_solution_long)



using JLD2
@save "knowledge_enhanced_NN.jld2" solution recovered_sys approximate_solution_long NNsolution L true_solution_long Xₙ L̄ L̂
@load "knowledge_enhanced_NN.jld2" solution recovered_sys approximate_solution_long NNsolution L true_solution_long Xₙ L̄ L̂

c1 = 3 # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple # RGBA(153/255,50/255,204/255,1) # Purple

p1 = plot(0.1:0.1:tspan[end],abs.(Array(solution)[:,2:end] .- NNsolution[:,2:end])' .+ eps(Float32),
          lw = 3, yaxis = :log, title = "Timeseries of UODE Error",
          color = [3 :orange], xlabel = "t",
          label = ["x(t)" "y(t)"],
          titlefont = "Helvetica", legendfont = "Helvetica",
          legend = :topright)

# Plot L₂
p2 = plot(X[1,:], X[2,:], L̂[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = c1,
     label = "Neural Network", xaxis = "x", yaxis="y",
     titlefont = "Helvetica", legendfont = "Helvetica",
     legend = :bottomright)
plot!(X[1,:], X[2,:], L̄[2,:], lw = 3, label = "True Missing Term", color=c2)

p3 = scatter(solution, color = [c1 c2], label = ["x data" "y data"],
             title = "Extrapolated Fit From Short Training Data",
             titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

plot!(p3,true_solution_long, color = [c1 c2], linestyle = :dot, lw=5, label = ["True x(t)" "True y(t)"])
plot!(p3,approximate_solution_long, color = [c3 c4], lw=1, label = ["Estimated x(t)" "Estimated y(t)"])
plot!(p3,[2.99,3.01],[0.0,maximum(hcat(Array(solution),Array(approximate_solution_long)))],lw=2,color=:black)
annotate!([(1.5,9,text("Training \nData", 10, :center, :top, :black, "Helvetica"))])
l = @layout [grid(1,2)
             grid(1,1)]
plot(p1,p2,p3,layout = l)

savefig("sindy_extrapolation.pdf")
