using NeuralNetDiffEq, StochasticDiffEq, Flux, LinearAlgebra, Statistics

d = 100 # number of dimensions
x0 = fill(0.0f0,d)
tspan = (0.0f0, 1.0f0)
dt = 0.2
m = 100 # number of trajectories (batch size)
λ = 1.0f0
#
g(X) = log(0.5f0 + 0.5f0*sum(X.^2))
f(X,u,σᵀ∇u,p,t) = -λ*sum(σᵀ∇u.^2)
μ(X,p,t) = -10.0f0.*(X./X)  #Vector d x 1 λ
σ(X,p,t) = Diagonal(sqrt(2.0f0)*ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ, σ, x0, tspan)

hls = 10 + d #hidden layer size
opt = Flux.ADAM(0.01)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)
#

@time ans = solve(prob, pdealg, verbose=true, maxiters=100, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-2)

@time ans = solve(prob, pdealg, verbose=true, maxiters=100, trajectories=m,
                            alg=LambaEM(), dt=dt, pabstol = 1f-2)

T = tspan[2]
MC = 10^5
W() = randn(d,1)
u_analytical(x, t) = -(1/λ)*log(mean(exp(-λ*g(x .+ sqrt(2.0)*abs.(T-t).*W())) for _ = 1:MC))
analytical_ans = u_analytical(x0, tspan[1])

error_l2 = sqrt((ans - analytical_ans)^2/ans^2)

println("Hamilton Jacobi Bellman Equation")
# println("numerical = ", ans)
# println("analytical = " , analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.2
