cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DiffEqFlux, Flux
using DifferentialEquations
using Plots, DelimitedFiles, Statistics
using Sundials

function FENEP!(out,du,u,p,t,γd)
  # DAE definition for FENE-p
  θ₁₁,θ₂₂,θ₁₂, τ₁₁,τ₂₂,τ₁₂ = u
  λ,η,L = p
  a = L^2 /(L^2 -3)
  fλ = (L^2 + (λ/η/a)*(τ₂₂+τ₁₁))/(L^2 - 3)
  out[1] =  τ₁₁ + du[1] - 2*λ*γd(t)*τ₁₂/fλ
  out[2] =  τ₂₂ + du[2]
  out[3] =  τ₁₂ + du[3] - λ*γd(t)*τ₂₂/fλ - η/fλ * γd(t)

  out[4] = θ₁₁ - λ*τ₁₁/fλ
  out[5] = θ₂₂ - λ*τ₂₂/fλ
  out[6] = θ₁₂ - λ*τ₁₂/fλ
end

function find_σ_exact(tsave,γd)
  # finds the exact solution of the FENE-p equations given a strain rate fn and
  # time points to save at
  λ = 2.0
  L = 2.0
  η = 4.0
  p = [λ,η,L]
  u₀ = zeros(6)
  du₀ = [0.0, 0.0, η*γd(0.0)*(L^2-3)/L^2, 0.0,0.0,0.0]
  tspan = (Float64(tsave[1]),Float64(tsave[end]))
  differential_vars = [true,true,true,false,false,false]
  prob = DAEProblem((out,du,u,p,t) -> FENEP!(out,du,u,p,t,γd),du₀,u₀,tspan,
                    p=p,differential_vars=differential_vars)
  sol = solve(prob,IDA(),saveat=Float64.(tsave))
  return [Float32(σ[6]) for σ in sol.u]
end

function mode_loss(f0,f1,γd)
    # Given the NN f0 and f1 and strain rate γd, this returns a function
    # which when evaluated gives the loss.
    tspan = (0.0f0,6.2831f0)
    tsave = range(tspan[1],tspan[2],length=100)
    σ_exact = find_σ_exact(tsave,γd)
    p = []
    σ0 =0.0f0
    u0 = Tracker.param([σ0])

    dudt_(u::TrackedArray,p,t) = f1(Flux.Tracker.collect(vcat(u, [γd(t)])))
    dudt_(u::AbstractArray,p,t) = Flux.data(f1(vcat(u,[γd(t)])))

    prob = ODEProblem(dudt_,u0,tspan,p)
    loss_rd() =   begin
        P_RD = vcat(Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=u0,saveat=tsave,
                                                   abstol=1e-6,reltol=1e-6))
                      ,γd.(tsave)')
        σ_out = [f0(P_RD[:,i])[1] for i = 1:size(P_RD,2)]
        return mean( (σ_out .- σ_exact).^2 )
    end
    return loss_rd
end

function test_NN(γd,f0,f1)
    # Test the trained NN against a new strain rate
    tspan = (0.0f0,10.0f0)
    tsave = range(tspan[1],tspan[2],length=100)
    σ_exact = find_σ_exact(tsave,γd)
    dudt_(u::AbstractArray,p,t) = Flux.data(f1(vcat(u,[γd(t)])))
    prob = ODEProblem(dudt_,[0.0f0],tspan,p)
    ode_solve = solve(prob,u0=[0.0f0],Tsit5(),abstol=1e-6,reltol=1e-6)
    f0_ = Flux.data(f0)
    σ_approx = [f0_([ode_solve(t)[1],γd(t)]).data[1] for t in tsave]

    return tsave, σ_approx, σ_exact
end
function test_err(γd,f0,f1)
    tsave, σ_approx, σ_exact = test_NN(γd,f0,f1)
    return mean( (σ_approx .- σ_exact).^2 )
end

## --- Define NN and linear model ---
f0_n = Chain(Dense(2,2,tanh), Dense(2,1))
f1_n = Chain(Dense(2,2,tanh), Dense(2,1))
f0_l = Chain(Dense(2,1))
f1_l = Chain(Dense(2,1))
# --- Define the total loss across various strain rates ----
t_loss(f0,f1) =  sum(mode_loss(f0,f1,t -> 12.0f0*cos.(ω.*t))() for ω in 1.0f0:0.2f0:2.0f0)
data = Iterators.repeated((), 1000)
opt = Descent(0.005)
p = []
# --- Define test strain rate here, so call back can see how all errors improve
#     with time ----
γd_test = t -> 12f0*cos.(1.5f0.*t) #+ 7f0*cos.(2.5f0.*t)
err_l = []
err_n = []
cb(err,f0,f1) = push!(err,[t_loss(f0,f1).data,test_err(γd_test,f0,f1)])
Flux.train!(() -> t_loss(f0_n,f1_n), params(f0_n,f1_n,p), data, opt,cb= () -> cb(err_n,f0_n,f1_n))
Flux.train!(() -> t_loss(f0_l,f1_l), params(f0_l,f1_l,p), data, opt,cb= () -> cb(err_l,f0_l,f1_l))

# --- Make plot of how error goes with time ------
Er = (i,er) -> [e[i] for e in er]
plot(Er(1,err_n),xscale=:log10,yscale=:log10,ylabel="Error",xlabel="Training steps",
        label="Training error, Neural net")
plot!(Er(2,err_n),label="Testing error, Neural net")
plot!(Er(1,err_l), label="Training error, linear model")
plot!(Er(2,err_l),label="Testing error, linear model")

# --- Plot approximations vs true sol ----
tsave, σ_approx_n, σ_exact = test_NN(γd_test,f0_n,f1_n)
tsave, σ_approx_l, σ_exact = test_NN(γd_test,f0_l,f1_l)
plot(tsave,σ_approx_n,m=:circle,label="NN solution",ylabel="stress",xlabel="time")
plot!(tsave,σ_approx_l,m=:hexagon,label="Linear model")
plot!(tsave,σ_exact,label="True solution",leg=:bottomright,size=(700,400))


# program takes time to run, so save data for tweaking figures
open("er_data.txt", "w") do io # or other path
           writedlm(io, [Er(1,err_n) Er(2,err_n) Er(1,err_l) Er(2,err_l)], ',')
end;

open("plt_data.txt", "w") do io # or other path
           writedlm(io, [[t for t in tsave]  σ_approx_n σ_approx_l σ_exact], ',')
end;

# Timing
using StaticArrays, BenchmarkTools
W1 = convert(SMatrix{2,2},Tracker.data(f1_n[1].W))
b1 = convert(SVector{2},Tracker.data(f1_n[1].b))
W2 = convert(SMatrix{1,2},Tracker.data(f1_n[2].W))
b2 = convert(SVector{1},Tracker.data(f1_n[2].b))

tspan = (0.0f0,10.0f0)
tsave = range(tspan[1],tspan[2],length=100)
σ_exact = find_σ_exact(tsave,γd_test)
dudt_(u,p,t) = first(W2*broadcast(tanh,W1*@SVector([u[1],γd_test(t)])+b1)+b2)
prob = ODEProblem(dudt_,0.0f0,tspan,p)
@btime ode_solve = solve(prob,Tsit5(),tsave=tsave)
@btime find_σ_exact(tsave,γd_test)
