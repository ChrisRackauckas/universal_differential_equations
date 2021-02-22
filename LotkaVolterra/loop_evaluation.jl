cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqFlux, Flux

using JLD2, FileIO
using Statistics, StatsBase
# Set a random seed for reproduceable behaviour
using StatsPlots
using Plots
gr()

function retrieve_results(i::Int64, fname::String = "Scenario_1_recovery_loop.jld2")
    try
        jldopen(fname, "r") do file
            res = file["$i"]["result"]
            losses = file["$i"]["losses"]
            return res, losses
        end
    catch
        @info "Failed on $i"
        return
    end
end

function check_eqs(Ψ::SparseIdentificationResult)
    sum(get_sparsity(Ψ)) != 2 && return false
    ps = parameters(Ψ.equations)
    xs = variables(Ψ.equations)
    return all(isequal.([ps[1]*xs[1]*xs[2]; ps[2]*xs[1]*xs[2]], [x.rhs for x in Ψ.equations.eqs]))
end

function collect_results(rng = 1:1:500)
    founds = zeros(Bool, length(rng))
    sparsities = zeros(Float64, length(rng))
    errors = zeros(Float64, length(rng))
    aiccs = zeros(Float64, length(rng))
    losses = Array{AbstractVector}(undef, length(rng))
    failed = zeros(Bool, length(rng))
    for (j,i) in enumerate(rng)
        exp_result = retrieve_results(i)
        if isnothing(exp_result)
            sparsities[j] = Inf
            aiccs[j] = Inf
            errors[j] = Inf
            losses[j] = [Inf]
            failed[j] = true
            continue
        end
        sparsities[j] = sum(get_sparsity(exp_result[1]))
        errors[j] = norm(get_error(exp_result[1]), 2)
        aiccs[j] = norm(get_aicc(exp_result[1]), 2)
        founds[j] = check_eqs(exp_result[1])
        losses[j] = exp_result[2]
    end
    return founds, sparsities, errors, aiccs, losses, failed
end

function return_trajectory_and_parameter(i::Int64, fname::String = "Scenario_1_recovery_loop.jld2")
    try
        jldopen(fname, "r") do file
            X = file["$i"]["X"]
            t = file["$i"]["t"]
            ps = file["$i"]["trained_parameters"]
            return X, t, ps
        end
    catch
        @info "Failed on $i"
        return
    end
end

rbf(x) = exp.(-(x.^2))

function build_estimate(i::Int64,fname::String = "Scenario_1_recovery_loop.jld2")
    X, t, ps = return_trajectory_and_parameter(i, fname)
    p_ = Float32[1.3, 0.9, 0.8, 1.8]
    U = FastChain(
        FastDense(2, 5, rbf),
        FastDense(5, 5, rbf),
        FastDense(5, 5, rbf),
        FastDense(5, 2),
    )

    # Define the hybrid model
    function ude_dynamics!(du, u, p, t)
        û = U(u, p) # Network prediction
        du[1] = p_[1] * u[1] + û[1]
        du[2] = -p_[4] * u[2] + û[2]
    end

    prob_nn = ODEProblem(ude_dynamics!, X[:, 1], (t[1], t[end]), ps)
    sol = solve(prob_nn, Tsit5(), saveat = t)
    return X, t, sol
end

## Collect for each noise lvl
res_1 = collect_results(1:1:100)
res_2 = collect_results(101:1:200)
res_3 = collect_results(201:1:300)
res_4 = collect_results(301:1:400)
res_5 = collect_results(401:1:500)
# As a matrix
founds = [res_1[1] res_2[1] res_3[1] res_4[1] res_5[1]]
sparsities = [res_1[2] res_2[2] res_3[2] res_4[2] res_5[2]]
errors = [res_1[3] res_2[3] res_3[3] res_4[3] res_5[3]]
aiccs = [res_1[4] res_2[4] res_3[4] res_4[4] res_5[4]]
losses =  [res_1[5] res_2[5] res_3[5] res_4[5] res_5[5]]
failed = [res_1[6] res_2[6] res_3[6] res_4[6] res_5[6]]

sum(failed)
## Get the colorpalette
cpal = palette(:Dark2_5)

## Found Equations plot
p_founds = bar([sum(fi) for fi in eachcol(founds)], ylim = (0, 100), label = nothing,
    xticks = (1:1:5, [1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2]),
    #color = [:green, :seagreen, :limegreen, :springgreen, :aquamarine],
    color = [cpal[i] for i in 1:5],
    xlabel = "Noise Lvl.", ylabel = "Recovered Equations")

savefig(p_founds, joinpath(pwd(), "plots", "Found_Equations_Loop.pdf"))

[sum(fi) for fi in eachcol(founds)] / 100
mean([sum(fi) for fi in eachcol(founds)])
std([sum(fi) for fi in eachcol(founds)])
## Complexity Plot

#marginalkde(errors[founds], aiccs[founds], xlabel = "L2-Error", ylabel = "AICC",
#    label = ["Noise Lvl. $i" for i in [1e-3 5e-3 1e-2 2.5e-2 5e-2]],
#    #color = [:green :seagreen :limegreen :springgreen :aquamarine], alpha = 0.3,
#    palette = :Dark2_5, alpha = 0.5,)

e_s = [e[f] for (e,f) in zip(eachcol(errors), eachcol(founds))]
a_s = [e[f] for (e,f) in zip(eachcol(aiccs), eachcol(founds))]

scatter(e_s, a_s, xlabel = "L2-Error", ylabel = "AICC",
        label = ["Noise Lvl. $i" for i in [1e-3 5e-3 1e-2 2.5e-2 5e-2]],
        #color = [:green :seagreen :limegreen :springgreen :aquamarine], alpha = 0.3,
        palette = :Dark2_5, alpha = 0.5,)

boxplot(e_s, palette = :Dark2_5, label = nothing, xlabel = "Noise Lvl.", ylabel = "L2-Error",
    xticks = (1:1:5, [1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2]),)
boxplot(a_s, palette = :Dark2_5, label = nothing, xlabel = "Noise Lvl.", ylabel = "AICC",
    xticks = (1:1:5, [1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2]),)

## Loss Plot
function loss_plot(losses)
    p = plot(ylabel = "Loss", xlabel = "Iterations")
    vline!([200], color = :black, lw = 1, style = :dot, label = nothing)

    lsucc = false
    lfailed = false

    for (i,l) in enumerate(losses)
        if failed[i]
            continue
        end
        indicator = founds[i] ? :green : :red

        if !lsucc && founds[i]
            label = "Successful"
            lsucc = true
        elseif !lfailed && !founds[i]
            label = "Failed"
            lfailed = true
        else
            label = nothing
        end
        scatter!([length(l)], [l[end]], color = indicator, alpha = 0.5, label = label, msize = 4.)
        plot!(l, color = :blue, alpha = 0.1, yaxis = :log10, label = nothing)
    end
    finite_losses = [findmax(l) for l in losses if length(l) > 1]
    lmax, _ = findmax(finite_losses)
    #display(p)
    annotate!([(100, lmax[1]*1.5, text("ADAM", :center))])
    annotate!([(800, lmax[1]*1.5, text("BFGS", :center))])
    return p
end

plosses = loss_plot(losses)
minlength, _ = findmin(length.(l for l in losses[founds]))
l_mins = hcat([l[1:minlength] for l in losses[founds]])
l_mean = mean(l_mins, dims = 2)
plot!(l_mean, color = :red, label = "Mean Loss")
savefig(plosses, joinpath(pwd(), "plots", "Losses.pdf"))
## Sample some results
neg_samples = collect(1:500)[vcat(.!founds...)]
neg_res = sample(neg_samples, 8,replace = false)
function plot_examples(inds, samplesize = 8, idxs = 1:500)
   _samples = sample(collect(idxs)[vcat(inds...)], samplesize, replace = false, ordered = true)
   p = []
   for s in _samples
       r_  = build_estimate(s)
       if isnothing(r_)
           continue
       end
       X, t, sol = r_
       p_ = plot(sol, color = [:red :blue],label = nothing)
       scatter!(t, X', color = [:red :blue], label = nothing)
       annotate!([(2, 4.3, text("Sample $s", :left))])
       push!(p, p_)
   end
   return p
end

n_pos = plot_examples(.!founds[end-200:10:end], 8, 300:10:500)
n_examples = plot(n_pos..., layout = (4,2), size = (800, 1000))
savefig(n_examples, joinpath(pwd(), "plots", "Failed_Examples.pdf"))
p_pos = plot_examples(founds[end-300:end], 8, 200:500)
p_examples = plot(p_pos..., layout = (4,2), size = (800, 1000))
savefig(p_examples, joinpath(pwd(), "plots", "Sucessful_Examples.pdf"))
