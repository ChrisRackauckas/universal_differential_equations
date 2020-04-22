cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

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

# Create a problem and a trajectory
prob = ODEProblem(lotka, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

scatter(solution, alpha = 0.25)
plot!(solution, alpha = 0.5)

# Ideal data
tsdata = Array(solution)
# Add noise to the data
noisy_data = tsdata + Float32(1e-3)*randn(eltype(tsdata), size(tsdata))

# Define the neueral network which learns L(x, y, y(t-τ))
# Actually, we do not care about overfitting right now, since we want to
# extract the derivative information without numerical differentiation.
ann = FastChain(FastDense(2, 64, swish),FastDense(64, 64, swish), FastDense(64, 64, swish),FastDense(64, 2))
p = initial_params(ann)

# Hybrid ODE
function dudt_(u, p,t)
    x, y = u
    z = ann(u,p)
    [p_[1]*x + z[1],
    -p_[4]*y + z[2]]
end

prob_nn = ODEProblem(dudt_,u0, tspan, p)

function predict(θ)
    Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = solution.t,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, noisy_data .- pred), pred # + 1e-5*sum(sum.(abs, params(ann)))
end

losses = []

callback(losses, θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
end

cb_(θ, l, pred) = callback(losses, θ, l, pred)

res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=cb_, maxiters = 100)
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=cb_, maxiters = 10000)

# Plot the losses
plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")

# Plot the data and the approximation
NNsolution = predict(res2.minimizer)
# Trained on noisy data vs real solution
plot(solution.t, NNsolution')
plot!(solution.t, tsdata')

# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[1]
for i ∈ 1:8
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:7
        if i != j
            push!(polys, u[1]^j*u[2]^i)
            push!(polys, u[2]^i*u[1]^i)
        end
    end
end

# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
basis = Basis(h, u)
println(basis)
# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = exp10.(-10:0.1:10)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)

# Test on original data and without further knowledge
X = Array(solution)
DX = Array(solution(solution.t, Val{1}))
Ψ = SInDy(X, DX[:, :], basis, λ, opt = opt, maxiter = 1000, f_target = f_target) # Fail
println(Ψ)
print_equations(Ψ)

# Test on ideal derivative data ( not available )
X = Array(solution)
L = [-p_[2]*(X[1,:].*X[2,:])';p_[3]*(X[1,:].*X[2,:])']
Ψ = SInDy(X, L, basis, λ, opt = opt, maxiter = 1000, f_target = f_target, denoise = true) # Succeed
println(Ψ)
print_equations(Ψ)

# Create the UODE data
prob_nn2 = ODEProblem(dudt_,u0, tspan, res2.minimizer)
_sol = solve(prob_nn2, Tsit5(), saveat = 0.01)
X̂ = Array(_sol)
L̂ = ann(_sol[:,:], res2.minimizer)
opt = STRRidge()
# Test on uode
Ψ = SInDy(X̂, L̂, basis, λ, opt = opt, maxiter = 100, denoise = false, normalize = true, f_target = f_target) # Succeed
println(Ψ)
print_equations(Ψ)
p̂ = parameters(Ψ)

# The parameters are a bit off, so we reiterate another sindy term to get closer to the ground truth

# Create function
unknown_sys = ODESystem(Ψ)
unknown_eq = ODEFunction(unknown_sys)

# Build a ODE for the estimated system
function approx(du, u, p, t)
    # Add SInDy Term
    α, δ, β, γ = p
    z = unknown_eq(u, [β; γ], t)
    du[1] = α*u[1] + z[1]
    du[2] = -δ*u[2] + z[2]
end

# Create the approximated problem and solution
ps = [p_[[1,4]]; p̂]
a_prob = ODEProblem(approx, u0, tspan, ps)
a_solution = solve(a_prob, Tsit5(), saveat = 0.1)

# Plot
plot(solution)
plot!(a_solution)


# Look at long term prediction
t_long = (0.0, 50.0)
a_prob = ODEProblem(approx, u0, t_long, ps)
a_solution = solve(a_prob, Tsit5()) # Using higher tolerances here results in exit of julia
plot(a_solution)

prob_true2 = ODEProblem(lotka, u0, t_long, p_)
solution_long = solve(prob_true2, Tsit5(), saveat = a_solution.t)
plot!(solution_long)



using JLD2
@save "knowledge_enhanced_NN.jld2" solution unknown_sys a_solution NNsolution ann solution_long X L X̂ L̂
@load "knowledge_enhanced_NN.jld2" solution unknown_sys a_solution NNsolution ann solution_long X L X̂ L̂

p1 = plot(0.1:0.1:tspan[end],abs.(Array(solution)[:,2:end] .- NNsolution[:,2:end])' .+ eps(Float32),
          lw = 3, yaxis = :log, title = "Timeseries of UODE Error",
          color = [3 :orange], xlabel = "t",
          label = ["x(t)" "y(t)"],
          titlefont = "Helvetica", legendfont = "Helvetica",
          legend = :topright)

# Plot L₂
p2 = plot(X[1,:], X[2,:], L[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = 3,
     label = "True Missing Term", xaxis = "x", yaxis="y",
     titlefont = "Helvetica", legendfont = "Helvetica",
     legend = :bottomright)
scatter!(X̂[1,1:5:end], X̂[2,1:5:end], L̂[2,1:5:end], lw = 3, label = "Neural Network Fit", color=:orange)

c1 = 3 # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple # RGBA(153/255,50/255,204/255,1) # Purple

p3 = scatter(solution, color = [c1 c2], label = ["x data" "y data"],
             title = "Extrapolated Fit From Short Training Data",
             titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

plot!(p3,solution_long, color = [c1 c2], linestyle = :dot, lw=5, label = ["True x(t)" "True y(t)"])
plot!(p3,a_solution, color = [c3 c4], lw=1, label = ["Estimated x(t)" "Estimated y(t)"])
plot!(p3,[2.99,3.01],[0.0,maximum(hcat(Array(solution),Array(a_solution)))],lw=2,color=:black)
annotate!([(1.5,9,text("Training \nData", 10, :center, :top, :black, "Helvetica"))])
l = @layout [grid(1,2)
             grid(1,1)]
plot(p1,p2,p3,layout = l)

savefig("sindy_extrapolation.pdf")
