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
p = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka, u0,tspan, p)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

scatter(solution, alpha = 0.25)
plot!(solution, alpha = 0.5)

tsdata = Array(solution)

# Define the neueral network which learns L(x, y, y(t-τ))
# Actually, we do not care about overfitting right now, since we want to
# extract the derivative information without numerical differentiation.
ann = FastChain(FastDense(2, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 2))
p = initial_params(ann)

function dudt_(u, p,t)
    x, y = u
    z = ann(u,p)
    [p[1]*x + z[1],
    -p[4]*y + z[2]]
end

prob_nn = ODEProblem(dudt_,u0, tspan, p)
s = concrete_solve(prob_nn, Tsit5(), u0, p, saveat = 0.1)

plot(solution)
plot!(s)

function predict(θ)
    Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = 0.1,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, tsdata .- pred), pred # + 1e-5*sum(sum.(abs, params(ann)))
end
loss(p)

const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
end

res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 100)
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)

# Plot the data and the approximation
NNsolution = predict(res2.minimizer)
plot(solution.t, NNsolution')
plot!(solution.t, tsdata')

# Collect the state trajectory and the derivatives
X = tsdata
DX = Array(solution(solution.t, Val{1})) #- [p[1]*(X[1,:])';  -p[4]*(X[2,:])']
L̃ = ann(X,p)

prob_nn2 = ODEProblem(dudt_,u0, tspan, res2.minimizer)
_sol = solve(prob_nn2, Tsit5())
DX_ = Array(_sol(solution.t, Val{1}))

# The learned derivatives
plot(DX')
plot!(DX_')

L = [-p[2]*(X[1,:].*X[2,:])';p[3]*(X[1,:].*X[2,:])']
L̂ = ann(X,res2.minimizer)
scatter(L̂')
plot(L')
plot!(L̃')

# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = []
for i ∈ 1:3
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:3
        if i == j
            push!(polys, (u[1]^i)*(u[2]^j))
        end
    end
end

# And some other stuff
h = [cos(u[1]); sin(u[1]); 1u[1]^0; polys...]

basis = Basis(h, u)
Ψ = SInDy(X, DX, basis) # Fail
println(Ψ.basis)
Ψ = SInDy(X, L, basis) # Suceed
println(Ψ.basis)
Ψ = SInDy(X, L̃, basis) # Fail
println(Ψ.basis)

# Works most of the time
θ = hcat([basis(xi, p = []) for xi in eachcol(X)]...)
Ξ = DataDrivenDiffEq.STRridge(θ', L̃', ϵ = 6e-1, maxiter = 1000)

# Derive a new basis with only structural parameters
Ξ2 = similar(Ξ)
p2 = p[[1, 4]]
for i in 1:size(Ξ, 2)
    for k in 1:size(Ξ, 1)
        if abs(Ξ[k, i]) > 0
            Ξ2[k, i] = 1.0
            push!(p2, Ξ[k, i])
        else
            Ξ2[k, i] = 0.0
        end

    end
end

Ψ = Basis(simplify_constants.(Ξ2'*basis.basis), u)
p2_ = Flux.param(p2)

function approx(du, u, p, t)
    # Add SInDy Term
    z = Ψ(u)
    du[1] = p[1]*u[1] + p[3]*z[1]
    du[2] = -p[2]*u[2] + p[4]*z[2]
end

a_prob = ODEProblem(approx, Float32.(u0), tspan, p2)
a_solution = solve(a_prob, Tsit5(), saveat = 0.1f0)

plot(a_solution)

function predict_adjoint()
    diffeq_adjoint(p2_, a_prob, Vern7(), saveat = solution.t)
end

predict_adjoint()[:,1:end-1]

loss_parameters() = sum(abs2, solution[:,:] - predict_adjoint()[:,1:end-1])

opt_parameters = ADAM(1e-2)
data = Iterators.repeated((), 1)
cb_parameters() = println(loss_parameters())
Flux.@epochs 600 Flux.train!(loss_parameters, Flux.params(p2_), data, opt_parameters, cb = cb_parameters)

# Make the plots
tspan = (0.0f0, 20.0f0)
a_prob = ODEProblem(approx, Float32.(u0), tspan, Flux.data(p2_))
a_solution = solve(a_prob, Vern7(), abstol=1e-8, reltol=1e-8, saveat = 0.1)
plot(a_solution)

prob_true2 = ODEProblem(lotka, u0,tspan, p)
solution_long = solve(prob_true2, Vern7(), abstol=1e-8, reltol=1e-8, saveat = 0.1)

using JLD2
@save "knowledge_enhanced_NN.jld2" solution Ψ a_solution NNsolution ann solution_long X L L̃
@load "knowledge_enhanced_NN.jld2" solution Ψ a_solution NNsolution ann solution_long X L L̃

p1 = plot(0.1:0.1:3,abs.(Array(solution)[:,2:end] .- NNsolution[:,2:end])' .+ eps(Float32),
          lw = 3, yaxis = :log, title = "Timeseries of UODE Error",
          color = [3 :orange], xlabel = "t",
          label = ["x(t)" "y(t)"],
          titlefont = "Helvetica", legendfont = "Helvetica",
          legend = :topright)

# Plot L₂
p2 = plot(X[1,:], X[2,:], L[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = 3,
     label = "Neural Network", xaxis = "x", yaxis="y",
     titlefont = "Helvetica", legendfont = "Helvetica",
     legend = :bottomright)
plot!(X[1,:], X[2,:], L̃[2,:], lw = 3, label = "True Missing Term", color=:orange)

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
