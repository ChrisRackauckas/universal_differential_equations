cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Single experiment, move to ensemble further on
# Some good parameter values are stored as comments right now
# because this is really good practice

using JLD2
using LinearAlgebra
using Plots
gr()



@load "loop_runs.jld2" loop_losses loop_error loop_sparsity failures

p1 = plot(xlabel = "Iterations", ylabel = "Loss")

for l in loop_losses
    plot!(l, alpha = 0.35, yaxis = :log, legend = nothing)
end

savefig("sindy_training.pdf")

p2 = plot(xlabel = "Sparsity", ylabel = "L2-Error")
xticks!(0:1:2)

f1 = Array{Float32}(undef, 2, length(loop_error))
f2 = similar(f1)
i = 1

for (e_, s_) in zip(loop_error, loop_sparsity)
    global f1, f2, i
    f1[:, i] = [e_[1]; s_[1]]
    f2[:, i] = [e_[2]; s_[2]]
    i += 1
end

scatter!(f1[2, :], f1[1, :], label = "Equation 1", alpha = 0.5)
scatter!(f2[2, :], f2[1, :], label = "Equation 2", alpha = 0.5)

p = plot(p1, p2, title = "Loss and Regression Results")

savefig("sindy_loop_training.pdf")