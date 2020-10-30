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

A = [FourierBasis(n_weights)]
rx_nn = TensorLayer(A, 1)

#conv with bias with initial values as 1/dx^2
w_err = 0.0
init_w = reshape([1.1 -2.5 1.0], (3, 1, 1, 1))
diff_cnn_ = Conv(init_w, [0.], pad=(0,0,0,0))

#initialize D0 close to D/dx^2
D0 = [6.5]

p1 = 0.01randn(n_weights)
p2,re2 = Flux.destructure(diff_cnn_)
p = [p1;p2;D0]

function nn_ode(u,p,t)
    u_cnn_1   = [p[end-4] * u[end] + p[end-3] * u[1] + p[end-2] * u[2]]
    u_cnn     = [p[end-4] * u[i-1] + p[end-3] * u[i] + p[end-2] * u[i+1] for i in 2:Nx-1]
    u_cnn_end = [p[end-4] * u[end-1] + p[end-3] * u[end] + p[end-2] * u[1]]

    # Equivalent using Flux, but slower!
    #CNN term with periodic BC
    #diff_cnn_ = Conv(reshape(p[(end-4):(end-2)],(3,1,1,1)), [0.0], pad=(0,0,0,0))
    #u_cnn = reshape(diff_cnn_(reshape(u, (Nx, 1, 1, 1))), (Nx-2,))
    #u_cnn_1 = reshape(diff_cnn_(reshape(vcat(u[end:end], u[1:1], u[2:2]), (3, 1, 1, 1))), (1,))
    #u_cnn_end = reshape(diff_cnn_(reshape(vcat(u[end-1:end-1], u[end:end], u[1:1]), (3, 1, 1, 1))), (1,))

    [rx_nn([u[i]/π],p[1:length(p1)])[1] for i in 1:Nx] + p[end] * vcat(u_cnn_1, u_cnn, u_cnn_end)
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
    rx_nn_p, diff_cnn_, D0 = p[1:length(p1)],re2(p[(length(p1)+1):(end-1)]),p[end]
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
        rx_line = plot(u, rx_nn.([[elem/π] for elem in u],(rx_nn_p,)), label="NN")[1];
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
        rx_pred = reduce(vcat,rx_nn.([[elem/π] for elem in u],(rx_nn_p,)))
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
    res3 = DiffEqFlux.sciml_train(loss_rd, res2.minimizer, BFGS(initial_stepnorm=0.00001), cb=cb, maxiters = 1000, allow_f_increases=true)
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

# 3 parameters

Loss: 0.0094    D0: 5.7811 Weights:(1.1501,      -2.3004,       1.1503)         Sum: -0.0000
updating figure
237.483188 seconds (875.53 M allocations: 44.014 GiB, 3.10% gc time)

Loss: 0.0095    D0: 5.7752 Weights:(1.1496,      -2.2988,       1.1492)         Sum: 0.0000
updating figure
saved figure
240.444140 seconds (885.66 M allocations: 44.519 GiB, 3.11% gc time)

Loss: 0.0086    D0: 5.7983 Weights:(1.1519,      -2.3035,       1.1517)         Sum: -0.0000
updating figure
234.218061 seconds (917.69 M allocations: 46.116 GiB, 3.20% gc time)

Loss: 0.0090    D0: 5.1810 Weights:(1.2899,      -2.5798,       1.2899)         Sum: 0.0000
updating figure
232.159919 seconds (930.76 M allocations: 46.780 GiB, 3.25% gc time)

Loss: 0.0095    D0: 5.7827 Weights:(1.1524,      -2.3052,       1.1527)         Sum: -0.0000
updating figure
239.525352 seconds (953.27 M allocations: 47.905 GiB, 3.23% gc time)

# 5 parameters

Loss: 0.0016    D0: 5.8669 Weights:(1.0638,      -2.1274,       1.0636)         Sum: 0.0000
updating figure
243.928965 seconds (1.00 G allocations: 50.893 GiB, 3.40% gc time)

Loss: 0.0069    D0: 5.8545 Weights:(1.0472,      -2.0954,       1.0482)         Sum: -0.0000
updating figure
244.803636 seconds (1.01 G allocations: 51.578 GiB, 3.51% gc time)

Loss: 0.0069    D0: 5.8545 Weights:(1.0472,      -2.0954,       1.0482)         Sum: -0.0000
updating figure
240.307752 seconds (1.00 G allocations: 51.091 GiB, 3.43% gc time)

Loss: 0.0085    D0: 5.8546 Weights:(1.0782,      -2.1505,       1.0723)         Sum: 0.0000
updating figure
265.215108 seconds (1.14 G allocations: 57.873 GiB, 3.59% gc time)

Loss: 0.0039    D0: 5.8330 Weights:(1.0403,      -2.0784,       1.0382)         Sum: -0.0000
updating figure
246.503862 seconds (1.07 G allocations: 54.304 GiB, 3.59% gc time)

# 7 parameters

Loss: 0.0012    D0: 5.9098 Weights:(1.0402,      -2.0810,       1.0408)         Sum: -0.0000
updating figure
242.417536 seconds (1.14 G allocations: 56.570 GiB, 3.71% gc time)

Loss: 0.0055    D0: 5.9350 Weights:(1.0404,      -2.0825,       1.0421)         Sum: -0.0000
updating figure
252.128401 seconds (1.16 G allocations: 57.507 GiB, 3.88% gc time)

Loss: 0.0086    D0: 5.9322 Weights:(1.0501,      -2.1005,       1.0503)         Sum: -0.0001
updating figure
249.834412 seconds (1.14 G allocations: 56.479 GiB, 3.71% gc time)

Loss: 0.0083    D0: 5.8940 Weights:(1.0432,      -2.0869,       1.0437)         Sum: 0.0001
updating figure
255.021489 seconds (1.21 G allocations: 59.928 GiB, 3.92% gc time)

Loss: 0.0052    D0: 5.9229 Weights:(1.0481,      -2.0961,       1.0481)         Sum: 0.0000
updating figure
253.374001 seconds (1.23 G allocations: 60.636 GiB, 3.71% gc time)

# 15 parameters

Loss: 0.0032    D0: 6.1526 Weights:(1.0477,      -2.0939,       1.0462)         Sum: -0.0000
updating figure
267.148464 seconds (1.72 G allocations: 83.585 GiB, 4.00% gc time)

Loss: 0.0057    D0: 6.1925 Weights:(1.0960,      -2.1901,       1.0942)         Sum: 0.0000
updating figure
281.902850 seconds (1.91 G allocations: 92.720 GiB, 4.15% gc time)

Loss: 0.0043    D0: 6.1548 Weights:(1.0607,      -2.1218,       1.0611)         Sum: 0.0000
updating figure
262.655202 seconds (1.76 G allocations: 85.483 GiB, 4.01% gc time)

Loss: 0.0088    D0: 6.1617 Weights:(1.0751,      -2.1503,       1.0751)         Sum: -0.0001
updating figure
267.427766 seconds (1.80 G allocations: 87.616 GiB, 3.99% gc time)

Loss: 0.0095    D0: 6.1630 Weights:(1.0691,      -2.1416,       1.0724)         Sum: -0.0001
updating figure
269.782135 seconds (1.80 G allocations: 87.805 GiB, 4.00% gc time)
=#

x = [242.417536,252.128401,249.834412,255.021489,253.374001]
mean(x)
using Statistics
std(x)
