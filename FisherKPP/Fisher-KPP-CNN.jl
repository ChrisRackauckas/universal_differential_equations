cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

#This script simulates the Fisher-KPP equation and fits
#a neural PDE to the data with the growth (aka reaction) term replaced
#by a feed-forward neural network and the diffusion term with a CNN

using PyPlot, Printf
using LinearAlgebra
using DifferentialEquations
using Flux, DiffEqFlux
using Printf

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

#save_folder = "plots/12-15-rx-CNN-withBC-1"
save_folder = "data"

if isdir(save_folder)
    rm(save_folder, recursive=true)
end
mkdir(save_folder)

close("all")
figure()
plot(x, rho0)
gcf()

reaction(u) = r * u .* (1 .- u)

lap = diagm(0 => -2.0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2
#Periodic BC
lap[1,end] = 1.0/dx^2
lap[end,1] = 1.0/dx^2
#Neumann BC
#lap[1,2] = 2.0/dx^2
#lap[end,end-1] = 2.0/dx^2

function rc_ode(drho, rho, p, t)
    #finite difference
    drho .= D * lap * rho + reaction.(rho)
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
savefig(@sprintf("%s/true_solution.pdf", save_folder))
gcf()

#### Define neural net for reverse mode AD
n_weights = 10

#for the reaction term
rx_nn = Chain(Dense(1, n_weights, tanh),
                Dense(n_weights, 2*n_weights, tanh),
                Dense(2*n_weights, n_weights, tanh),
                Dense(n_weights, 1),
                x -> x[1])

#conv with bias with initial values as 1/dx^2
w_err = 0.0
init_w = reshape([1.1 -2.5 1.0], (3, 1, 1, 1))
sum(init_w)
diff_cnn_ = Conv(param(init_w), param([0.]), pad=(0,0,0,0))

#remove bias
diff_cnn(x) = diff_cnn_(x) .- diff_cnn_.bias

rx_nn_dat = Chain(rx_nn, x -> x.data)

#initialize D0 close to D/dx^2
D0 = param([6.5])

function nn_ode(u::TrackedArray,p,t)

    #CNN term with periodic BC
    u_cnn = reshape(diff_cnn(reshape(u, (Nx, 1, 1, 1))), (Nx-2,))
    u_cnn_1 = reshape(diff_cnn(reshape(vcat(u[end:end], u[1:1], u[2:2]), (3, 1, 1, 1))), (1,))
    u_cnn_end = reshape(diff_cnn(reshape(vcat(u[end-1:end-1], u[end:end], u[1:1]), (3, 1, 1, 1))), (1,))

    du = Tracker.collect(rx_nn.([u[i:i] for i in 1:Nx])) +
            p[1] * vcat(u_cnn_1, u_cnn, u_cnn_end)

    return du
end

function nn_ode(u::AbstractArray,p,t)
    du = [Flux.data(rx_nn([u[i]]))[1] for i in 1:Nx]
            + Flux.data(p)[1] * Flux.data(reshape(diff_cnn(u), (Nx,)))
    return du
end

#set up ODE problem
prob_nn = ODEProblem(nn_ode,param(rho0), (0.0, T), D0)
sol_nn = diffeq_rd(D0, prob_nn,Tsit5())

function predict_rd()
  Flux.Tracker.collect(diffeq_rd(D0,prob_nn,Tsit5(),u0=param(rho0),saveat=dt))
end

#match data and force the weights of the CNN to add up to zero
loss_rd() = sum(abs2, ode_data .- predict_rd()) + 10^2 * abs(sum(diff_cnn_.weight))

#Optimization
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
cb = function ()
    push!(train_arr, Flux.data(loss_rd()))
    push!(diff_arr, Flux.data(D0)[1])

    weight = diff_cnn_.weight[:].data
    push!(w1_arr, weight[1])
    push!(w2_arr, weight[2])
    push!(w3_arr, weight[3])

    println(@sprintf("Loss: %0.4f\tD0: %0.4f Weights:(%0.4f,\t %0.4f, \t%0.4f) \t Sum: %0.4f"
            , loss_rd(), Flux.data(D0)[1], weight[1], weight[2], weight[3], sum(weight)))

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
        cur_pred = Flux.data(predict_rd())
        img = pcolormesh(x,t,cur_pred')
        global img
        xlabel(L"$x$"); ylabel(L"$t$"); title("Prediction")
        #colb = colorbar(); global colb
        colorbar(); clim([0, 1]);

        ax = subplot(133); global ax
        u = collect(0:0.01:1)
        rx_line = plot(u, rx_nn_dat.([[elem] for elem in u]), label="NN")[1];
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
        cur_pred = Flux.data(predict_rd())
        img.set_array(cur_pred[1:end-1, 1:end-1][:])
        ttl.set_text(@sprintf("Epoch = %d", count))

        u = collect(0:0.01:1)
        rx_pred = rx_nn_dat.([[elem] for elem in u])
        rx_line.set_ydata(rx_pred)
        u = collect(0:0.01:1)

        min_lim = min(minimum(rx_pred), minimum(reaction.(u)))-0.1
        max_lim = max(maximum(rx_pred), maximum(reaction.(u)))+0.1

        ax.set_ylim([min_lim, max_lim])
        #colb.set_clim([minimum(cur_pred[:]), maximum(cur_pred[:])])
    end

    global save_count
    if count%save_freq == 0
        println("saved figure")
        savefig(@sprintf("%s/pred_%05d.png", save_folder, save_count), dpi=200, bbox_inches="tight")
        save_count += 1
    end

    display(gcf())
    count += 1

end
cb()

using Flux: @epochs
#copy script
cp("KPP-CNN-with-BC.jl", @sprintf("%s/runscript.jl", save_folder), force=true)
@epochs 1000 Flux.train!(loss_rd, params(rx_nn, diff_cnn_, D0), [()], opt, cb = cb)

## Save trained model
using BSON: @save, @load
@save @sprintf("%s/model.bson", save_folder) rx_nn diff_cnn_ w1_arr w2_arr w3_arr train_arr diff_arr

#plot for PNAS paper
@load @sprintf("%s/model.bson", save_folder) rx_nn diff_cnn_ w1_arr w2_arr w3_arr train_arr diff_arr
#re-defintions for newly loaded data
diff_cnn(x) = diff_cnn_(x) .- diff_cnn_.bias
rx_nn_dat = Chain(rx_nn, x -> x.data)
D0 = diff_arr[end]

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
    cur_pred = Flux.data(predict_rd())
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
    plot(u, rx_nn_dat.([[elem] for elem in u]), label="UPDE")[1];
    plot(u, reaction.(u), linestyle="--", label="True")
    xlabel(L"$\rho$")
    title("Reaction Term")
    legend(loc="lower center", frameon=false, fontsize=6);
    ylim([0, 0.3])

    tight_layout(h_pad=1)
    gcf()

savefig(@sprintf("%s/fisher_kpp.pdf", save_folder))

#save loss vs epochs plot
figure(figsize=(6,3))
plot(log.(train_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel("Log(loss)")
tight_layout()
savefig(@sprintf("%s/loss_vs_epoch.pdf", save_folder))
gcf()

#save D0 vs epochs plot
figure(figsize=(6,3))
plot(abs.(diff_arr)*dx^2, "k.", markersize=1)
xlabel("Epochs"); ylabel(L"$D$")
ylim([0.009, 0.011])
axhline(0.010, linestyle="--", color="k", label="Expected Value")
tight_layout()
legend(frameon=false)
savefig(@sprintf("%s/D0_vs_epoch.pdf", save_folder))
gcf()

#plot the loss and weights plot and save
figure(figsize=(8,2.5))

subplot(131)
plot(log.(train_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel("Log(loss)")

subplot(132)
plot(Flux.data(w1_arr ./ w3_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel(L"$w_1/w_3$")
axhline(1.0, linestyle="--", color="k", label="Expected Value")
legend(loc="upper right", frameon=false, fontsize=8)

subplot(133)
plot(Flux.data(w1_arr .+ w2_arr .+ w3_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel(L"$w_1 + w_2 + w_3$")
axhline(0.0, linestyle="--", color="k", label="Expected Value")
legend(loc="lower right", frameon=false, fontsize=8)
tight_layout()
gcf()

savefig(@sprintf("%s/weights_vs_epoch.pdf", save_folder))
