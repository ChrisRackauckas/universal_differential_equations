cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

#This script simulates the Fisher-KPP equation and fits
#a neural PDE to the data with the growth (aka reaction) term replaced
#by a feed-forward neural network and the diffusion term with a CNN

using PyPlot, Printf
using LinearAlgebra
using Flux, DiffEqFlux, Optim, DiffEqSensitivity
using BSON: @save, @load
using Flux: @epochs
using OrdinaryDiffEq

#parameter
D = 0.01; #diffusion
r = 1.0; #reaction rate

#domain
X = 1.0; T = 5;
dx = 0.04; dt = T/10;
x = collect(0:dx:X);
t = collect(0:dt:T);
Nx = Int64(X/dx+1);
Nt = Int64(T/dt+1);

#initial conditions
Amp = 1.0;
Delta = 0.2
#IC-1
rho0 = Amp*(tanh.((x .- (0.5 - Delta/2))/(Delta/10)) - tanh.((x .- (0.5 + Delta/2))/(Delta/10)))/2
#IC-2
#rho0 = Amp*(1 .- tanh.((x .- 0.2)/(Delta/6)))/2.

save_folder = "data"

if isdir(save_folder)
    rm(save_folder, recursive=true)
end
mkdir(save_folder)

close("all")
figure()
plot(x, rho0)
title("Initial Condition")
gcf()

########################
# Generate training data
########################
reaction(u) = r * u .* (1 .- u)
lap = diagm(0 => -2.0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2
#Periodic BC
lap[1,end] = 1.0/dx^2
lap[end,1] = 1.0/dx^2
#Neumann BC
#lap[1,2] = 2.0/dx^2
#lap[end,end-1] = 2.0/dx^2

function rc_ode(rho, p, t)
    #finite difference
    D * lap * rho + reaction.(rho)
end

prob = ODEProblem(rc_ode, rho0, (0.0, T), saveat=dt)
sol = solve(prob, Tsit5());
ode_data = Array(sol);

figure(figsize=(8,3))

subplot(121)
pcolor(x,t,ode_data')
xlabel("x"); ylabel("t");
colorbar()

subplot(122)
for i in 1:2:Nt
    plot(x, ode_data[:,i], label="t=$(sol.t[i])")
end
xlabel("x"); ylabel(L"$\rho$")
legend(frameon=false, fontsize=7, bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
tight_layout()
savefig(@sprintf("%s/training_data.pdf", save_folder))
gcf()

########################
# Define the neural PDE
########################
n_weights = 3

#for the reaction term
rx_nn = Chain(Dense(1, n_weights, tanh),
                Dense(n_weights, 1),
                x -> x[1])

#conv with bias with initial values as 1/dx^2
w_err = 0.0
init_w = reshape([1.1 -2.5 1.0], (3, 1, 1, 1))
diff_cnn_ = Conv(init_w, [0.], pad=(0,0,0,0))

#initialize D0 close to D/dx^2
D0 = [6.5]

p1,re1 = Flux.destructure(rx_nn)
p2,re2 = Flux.destructure(diff_cnn_)
p = [p1;p2;D0]
full_restructure(p) = re1(p[1:length(p1)]), re2(p[(length(p1)+1):end-1]), p[end]

function nn_ode(u,p,t)
    rx_nn = re1(p[1:length(p1)])

    u_cnn_1   = [p[end-4] * u[end] + p[end-3] * u[1] + p[end-2] * u[2]]
    u_cnn     = [p[end-4] * u[i-1] + p[end-3] * u[i] + p[end-2] * u[i+1] for i in 2:Nx-1]
    u_cnn_end = [p[end-4] * u[end-1] + p[end-3] * u[end] + p[end-2] * u[1]]

    # Equivalent using Flux, but slower!
    #CNN term with periodic BC
    #diff_cnn_ = Conv(reshape(p[(end-4):(end-2)],(3,1,1,1)), [0.0], pad=(0,0,0,0))
    #u_cnn = reshape(diff_cnn_(reshape(u, (Nx, 1, 1, 1))), (Nx-2,))
    #u_cnn_1 = reshape(diff_cnn_(reshape(vcat(u[end:end], u[1:1], u[2:2]), (3, 1, 1, 1))), (1,))
    #u_cnn_end = reshape(diff_cnn_(reshape(vcat(u[end-1:end-1], u[end:end], u[1:1]), (3, 1, 1, 1))), (1,))

    [rx_nn([u[i]])[1] for i in 1:Nx] + p[end] * vcat(u_cnn_1, u_cnn, u_cnn_end)
end

########################
# Soving the neural PDE and setting up loss function
########################
prob_nn = ODEProblem(nn_ode, rho0, (0.0, T), p)
sol_nn = concrete_solve(prob_nn,Tsit5(), rho0, p)

function predict_rd(θ)
  # No ReverseDiff if using Flux
  Array(concrete_solve(prob_nn,Tsit5(),rho0,θ,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

#match data and force the weights of the CNN to add up to zero
function loss_rd(p)
    pred = predict_rd(p)
    sum(abs2, ode_data .- pred) + 10^2 * abs(sum(p[end-4 : end-2])), pred
end

########################
# Training
########################

#Optimizer
opt = ADAM(0.001)

global count = 0
global save_count = 0
save_freq = 50

train_arr = Float64[]
diff_arr = Float64[]
w1_arr = Float64[]
w2_arr = Float64[]
w3_arr = Float64[]

#callback function to observe training
cb = function (p,l,pred)
    rx_nn, diff_cnn_, D0 = full_restructure(p)
    push!(train_arr, l)
    push!(diff_arr, p[end])

    weight = diff_cnn_.weight[:]
    push!(w1_arr, weight[1])
    push!(w2_arr, weight[2])
    push!(w3_arr, weight[3])

    println(@sprintf("Loss: %0.4f\tD0: %0.4f Weights:(%0.4f,\t %0.4f, \t%0.4f) \t Sum: %0.4f"
            ,l, D0[1], weight[1], weight[2], weight[3], sum(weight)))

    global count

    if count==0
        fig = figure(figsize=(8,2.5));
        ttl = fig.suptitle(@sprintf("Epoch = %d", count), y=1.05)
        global ttl
        subplot(131)
        pcolormesh(x,t,ode_data')
        xlabel(L"$x$"); ylabel(L"$t$"); title("Data")
        colorbar()

        subplot(132)
        img = pcolormesh(x,t,pred')
        global img
        xlabel(L"$x$"); ylabel(L"$t$"); title("Prediction")
        colorbar(); clim([0, 1]);

        ax = subplot(133); global ax
        u = collect(0:0.01:1)
        rx_line = plot(u, rx_nn.([[elem] for elem in u]), label="NN")[1];
        global rx_line
        plot(u, reaction.(u), label="True")
        title("Reaction Term")
        legend(loc="upper right", frameon=false, fontsize=8);
        ylim([0, r*0.25+0.2])

        subplots_adjust(top=0.8)
        tight_layout()
    end

    if count>0
        println("updating figure")
        img.set_array(pred[1:end-1, 1:end-1][:])
        ttl.set_text(@sprintf("Epoch = %d", count))

        u = collect(0:0.01:1)
        rx_pred = rx_nn.([[elem] for elem in u])
        rx_line.set_ydata(rx_pred)
        u = collect(0:0.01:1)

        min_lim = min(minimum(rx_pred), minimum(reaction.(u)))-0.1
        max_lim = max(maximum(rx_pred), maximum(reaction.(u)))+0.1

        ax.set_ylim([min_lim, max_lim])
    end

    global save_count
    if count%save_freq == 0
        println("saved figure")
        savefig(@sprintf("%s/pred_%05d.png", save_folder, save_count), dpi=200, bbox_inches="tight")
        save_count += 1
    end

    display(gcf())
    count += 1

    l < 0.01 # Exit when fit to 2 decimal places
end

#train
@time begin
    res1 = DiffEqFlux.sciml_train(loss_rd, p, ADAM(0.001), cb=cb, maxiters = 100)
    res2 = DiffEqFlux.sciml_train(loss_rd, res1.minimizer, ADAM(0.001), cb=cb, maxiters = 300)
    res3 = DiffEqFlux.sciml_train(loss_rd, res2.minimizer, BFGS(), cb=cb, maxiters = 1000, allow_f_increases=true)
end

pstar = res3.minimizer

## Save trained model
@save @sprintf("%s/model.bson", save_folder) pstar

########################
# Plot for paper
########################
@load @sprintf("%s/model.bson", save_folder) pstar
#re-defintions for newly loaded data

diff_cnn_ = Conv(reshape(pstar[(end-4):(end-2)],(3,1,1,1)), [0.0], pad=(0,0,0,0))
diff_cnn(x) = diff_cnn_(x) .- diff_cnn_.bias
D0 = res3.minimizer[end]

fig = figure(figsize=(4,4))

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["text.usetex"] = true
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = "Helvetica"
rcParams["axes.titlesize"] = 10

subplot(221)
pcolormesh(x,t,ode_data', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Data")
yticks([0, 1, 2, 3, 4, 5])

ax = subplot(222)
cur_pred = predict_rd(pstar)[1]
img = pcolormesh(x,t,cur_pred', rasterized=true)
global img
xlabel(L"$x$"); ylabel(L"$t$"); title("Prediction")
yticks([0, 1, 2, 3, 4, 5])
cax = fig.add_axes([.48,.62,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$\rho$")
clim([0, 1]);
colb.set_ticks([0, 1])

subplot(223)
plot(Flux.data(w1_arr ./ w3_arr) .- 1, label=L"$w_1/w_3 - 1$")
plot(Flux.data(w1_arr .+ w2_arr .+ w3_arr), label=L"$w_1 + w_2 + w_3$")
axhline(0.0, linestyle="--", color="k")
xlabel("Epochs"); title("CNN Weights")
xticks([0, 1500, 3000]); yticks([-0.4, -0.3,-0.2, -0.1, 0.0, 0.1])
legend(loc="lower right", frameon=false, fontsize=6)

subplot(224)
u = collect(0:0.01:1)
plot(u, rx_nn.([[elem] for elem in u]), label="UPDE")[1];
plot(u, reaction.(u), linestyle="--", label="True")
xlabel(L"$\rho$")
title("Reaction Term")
legend(loc="lower center", frameon=false, fontsize=6);
ylim([0, 0.3])

tight_layout(h_pad=1)
gcf()
savefig(@sprintf("%s/fisher_kpp.pdf", save_folder))

#plot loss vs epochs and save
figure(figsize=(6,3))
plot(log.(train_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel("Log(loss)")
tight_layout()
savefig(@sprintf("%s/loss_vs_epoch.pdf", save_folder))
gcf()


#=
# Success rate

# 2 decimal places 15 parameters

Loss: 0.0062    D0: 5.9522 Weights:(1.0385,      -2.0765,
    1.0380)          Sum: -0.0000                               1.0380)
updating figure
3430.385576 seconds (4.28 G allocations: 248.824 GiB, 1.31%
gc time)

Loss: 0.0095    D0: 6.1954 Weights:(0.9715,      -1.9432,
        0.9716)          Sum: -0.0000
updating figure
2824.449183 seconds (3.35 G allocations: 194.702 GiB, 1.32%
gc time)

Loss: 0.0094    D0: 5.7375 Weights:(1.0357,      -2.0714,
        1.0358)          Sum: -0.0000
updating figure
1174.592376 seconds (1.41 G allocations: 83.035 GiB, 1.22% gc time)

Loss: 0.0084    D0: 5.9525 Weights:(1.0049,      -2.0096,
        1.0047)          Sum: 0.0000
updating figure
1334.078451 seconds (1.61 G allocations: 94.637 GiB, 1.23% gc time)

Loss: 0.0075    D0: 7.4841 Weights:(0.8076,      -1.6145,       0.8069)          Sum: -0.0000
updating figure
saved figure
1053.729274 seconds (1.16 G allocations: 68.289 GiB, 1.12% gc time)

# 2 decimal places 7 parameters

Loss: 0.0095    D0: 6.1389 Weights:(0.9875,      -1.9749,
        0.9873)          Sum: 0.0000
updating figure
1415.089830 seconds (1.43 G allocations: 82.891 GiB, 1.37% gc time)

Loss: 0.0095    D0: 6.1381 Weights:(0.9447,      -1.8887,       0.9441)          Sum: -0.0000
updating figure
3293.038574 seconds (4.09 G allocations: 234.305 GiB, 1.10% gc time)

Loss: 0.0095    D0: 5.9216 Weights:(0.9869,      -1.9738,
        0.9869)          Sum: 0.0000
updating figure
3233.307375 seconds (4.09 G allocations: 234.477 GiB, 0.98%
gc time

Loss: 0.0095    D0: 5.9216 Weights:(0.9869,      -1.9738,
        0.9869)          Sum: 0.0000
updating figure
3265.894690 seconds (4.09 G allocations: 234.477 GiB, 0.98%
gc time)

Loss: 0.0093    D0: 6.4483 Weights:(0.9128,      -1.8256,       0.9128)         Sum: 0.0000
updating figure
1332.252349 seconds (1.44 G allocations: 83.367 GiB, 1.09% gc time)

# 2 decimal places 4 parameters

Loss: 0.4370    D0: 0.0457 Weights:(138.9983,    -277.9936,     138.9953)       Sum: 0.0000
updating figure
2210.916386 seconds (2.53 G allocations: 146.694 GiB, 1.07% gc time)

Loss: 0.3760    D0: 103.9210 Weights:(0.0533,    -0.1073,       0.0533)         Sum: -0.0007
updating figure
2296.853421 seconds (2.67 G allocations: 154.814 GiB, 1.08% gc time)

Loss: 0.3894    D0: 160.0180 Weights:(0.0367,    -0.0737,       0.0367)         Sum: -0.0003
updating figure
2262.799906 seconds (2.61 G allocations: 151.395 GiB, 1.08% gc time)

Loss: 0.3894    D0: 160.0180 Weights:(0.0367,    -0.0737,       0.0367)         Sum: -0.0003
updating figure
2346.022370 seconds (2.61 G allocations: 151.395 GiB, 1.09% gc time)

Loss: 0.2225    D0: 2739.4200 Weights:(0.0020,   -0.0040,       0.0020)         Sum: -0.0001
updating figure
5764.320007 seconds (6.49 G allocations: 372.446 GiB, 1.22% gc time)
=#

x = [1415.089830,3293.038574,3233.307375,3265.894690,1332.252349]
mean(x)
using Statistics
std(x)
