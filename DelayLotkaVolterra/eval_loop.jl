cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Single experiment, move to ensemble further on
# Some good parameter values are stored as comments right now
# because this is really good practice

using JLD2
using LinearAlgebra
using Statistics
using Plots
gr()



@load "loop_runs.jld2" loop_losses loop_error loop_sparsity failures

# Compute some Statistics over the training loops
l_ends = [l[end] for l in loop_losses]
l_length = [length(l) for l in loop_losses]
mean(l_ends)
std(l_ends)
minimum(l_length)
maximum(l_length)
# Failures in %
failures / length(loop_losses) * 100


p1 = plot(xlabel = "Iterations", ylabel = "Loss")

for l in loop_losses
    plot!(l, alpha = 0.35, yaxis = :log, legend = nothing)
end

display(p1)

savefig("sindy_training.pdf")

f1 = Array{Float32}(undef, 2, length(loop_error))
f2 = similar(f1)
f3 = similar(f1)
i = 1

for (e_, s_) in zip(loop_error, loop_sparsity)
    global f1, f2, i
    f1[:, i] = [e_[1]; s_[1]]
    f2[:, i] = [e_[2]; s_[2]]
    f3[:, i] = [sum(e_); loop_losses[i][end]]
    i += 1
end

p2 = scatter(f1[1, :], f2[1, :],
    xlabel = "L2 Error of Equation 1", ylabel = "L2 Error of Equation 2",
    alpha = 0.3, legend = nothing, xlim = (0, 0.5), ylim = (0, 0.5))

p3 = scatter(f3[1, : ], f3[2, :],
    ylabel = "Final Training Loss", xlabel = "Summed L2 Error of Model",
    legend = nothing, alpha = 0.3, yaxis = :log)


p = plot(p1, p3)

savefig("sindy_loop_training.pdf")
