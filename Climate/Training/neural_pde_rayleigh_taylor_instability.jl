using Printf
using Random
using Statistics
using LinearAlgebra

using Flux
using DifferentialEquations
using DiffEqFlux

using JLD2
using BSON
using Plots

using Flux: @epochs

#####
##### Load data from JLD2 file
#####

file = jldopen("../DataGeneration/rayleigh_taylor_instability_3d_horizontal_averages.jld2")

Is = keys(file["timeseries/t"])

Nz = file["grid/Nz"]
Lz = file["grid/Lz"]
Nt = length(Is)

t = zeros(Nt)
b = b_data = zeros(Nt, Nz)

for (i, I) in enumerate(Is)
    t[i] = file["timeseries/t/$I"]
    b[i, :] .= file["timeseries/b/$I"]
end

#####
##### Plot animation of b̅(z,t) from data
#####

z = file["grid/zC"]

@info "Animating buoyancy profile..."
anim = @animate for n=1:Nt
    t_str = @sprintf("%2.2f", t[n])
    plot(b[n, :], z, linewidth=2,
         xlim=(-1, 1), ylim=(-0.5, 0.5), label="",
         xlabel="buoyancy", ylabel="Depth (z)",
         title="Rayleigh-Taylor instability: t=$t_str", show=false)
end
mp4(anim, "rayleigh_taylor_instability_buoyancy.mp4", fps=15)

#####
##### Coarse grain data to 16 vertical levels (plus halo regions)
#####

function coarse_grain(data, resolution)
    @assert length(data) % resolution == 0
    s = length(data) / resolution
    
    data_cs = zeros(resolution)
    for i in 1:resolution
        t = data[Int((i-1)*s+1):Int(i*s)]
        data_cs[i] = mean(t)
    end
    
    return data_cs
end

coarse_resolution = cr = 16

b_cs = zeros(cr, Nt)
z_cs = coarse_grain(collect(z), cr)

for n=1:Nt
    b_cs[:, n] .= coarse_grain(b[n, :], cr)
end

#####
##### Plot coarse buoyancy profile
#####

@info "Animating coarse buoyancy profile..."
anim = @animate for n=1:Nt
    t_str = @sprintf("%2.2f", t[n])
    plot(b_cs[:, n], z_cs, linewidth=2,
         xlim=(-1, 1), ylim=(-1/2, 1/2), label="",
         xlabel="buoyancy", ylabel="Depth (z)",
         title="Rayleigh-Taylor instability: t=$t_str", show=false)
end
mp4(anim, "rayleigh_taylor_instability_buoyancy_coarse.mp4", fps=15)

#####
##### Create training data
#####

bₙ    = zeros(cr, Nt-1)
bₙ₊₁  = zeros(cr, Nt-1)

for i in 1:Nt-1
       bₙ[:, i] .=  b_cs[:,   i]
     bₙ₊₁[:, i] .=  b_cs[:, i+1]
end

N = 32  # Number of training data pairs.

training_data = [(bₙ[:, i], bₙ₊₁[:, i]) for i in 1:N]

#####
##### Create neural network
#####

# Complete black box right-hand-side.
dbdt_NN = Chain(Dense( cr, 2cr, relu),
                Dense(2cr, 4cr, relu),
                Dense(4cr, 2cr, relu),
                Dense(2cr,  cr))

NN_params = Flux.params(dbdt_NN)

#####
##### Define loss function
#####

tspan = (0.0, 0.1)  # 10 minutes
neural_pde_prediction(b₀) = neural_ode(dbdt_NN, b₀, tspan, Tsit5(), reltol=1e-4, save_start=false, saveat=tspan[2])

loss_function(bₙ, bₙ₊₁) = sum(abs2, bₙ₊₁ .- neural_pde_prediction(bₙ))

#####
##### Choose optimization algorithm
#####

opt = ADAM(0.1)

#####
##### Callback function to observe training.
#####

function cb()
    train_loss = sum([loss_function(bₙ[:, i], bₙ₊₁[:, i]) for i in 1:N])

    # nn_pred = neural_ode(dTdt_NN, bₙ[:, 1], (t[1], t[N]), Tsit5(), saveat=t[1:N], reltol=1e-4) |> Flux.data
    # test_loss = sum(abs2, b_cs[:, 1:N] .- nn_pred)
    
    println("train_loss = $train_loss")
    return train_loss
end

cb()

#####
##### Train!
#####

epochs = 10
best_loss = Inf
last_improvement = 0

for epoch_idx in 1:epochs
    global best_loss, last_improvement

    @info "Epoch $epoch_idx"
    Flux.train!(loss_function, NN_params, training_data, opt, cb=cb) # cb=Flux.throttle(cb, 10))
    
    loss = cb()

    if loss <= best_loss
        @info("Record low loss! Saving neural network out to dbdt_NN.bson")
        BSON.@save "dbdt_NN.bson" dbdt_NN
        best_loss = loss
        last_improvement = epoch_idx
    end
   
    # If we haven't seen improvement in 2 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 2 && opt.eta > 1e-6
        opt.eta /= 2.0
        @warn("Haven't improved in a while, dropping learning rate to $(opt.eta)")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end 
end

#####
##### Run the neural PDE forward to see how well it performs just by itself.
#####

nn_pred = neural_ode(dbdt_NN, bₙ[:, 1], (t[1], t[end]), Tsit5(), saveat=t, reltol=1e-4) |> Flux.data

z_cs = coarse_grain(z, cr)

anim = @animate for n=1:Nt
    t_str = @sprintf("%2.2f", t[n])
    plot(b_cs[:, n], z_cs, linewidth=2,
         xlim=(-1, 1), ylim=(-0.5, 0.5), label="Data",
         xlabel="buoyancy", ylabel="Depth (z)",
         title="Rayleigh-Taylor instability: t=$t_str",
         legend=:bottomright, show=false)
    plot!(nn_pred[:, n], z_cs, linewidth=2, label="Neural DE", show=false)
end

gif(anim, "rayleigh_taylor_instability_neural_PDE.gif", fps=15)

