#This script simulates the Fisher-KPP equation and fits
#a neural PDE to the data

using PyPlot, Printf

#parameters
D = 0.01; #diffusion
r = 1.0; #reaction rate

#domain
X = 1.0; T = 5.0;
dx = 0.04; dt = T/10;
x = collect(0:dx:X);
t = collect(0:dt:T);
Nx = Int64(X/dx+1);
Nt = Int64(T/dt+1);

#initial condition
Amp = 1.0;
Delta = 0.2
#IC-1
rho0 = Amp*(tanh.((x .- (0.5 - Delta/2))/(Delta/10)) - tanh.((x .- (0.5 + Delta/2))/(Delta/10)))/2
#IC-2
#rho0 = Amp*(1 .- tanh.((x .- 0.2)/(Delta/6)))/2.

#save_folder = "plots/12-12-rx-wave"
save_folder = "data"

if isdir(save_folder)
    rm(save_folder, recursive=true)
end
mkdir(save_folder)

figure()
plot(x, rho0)
gcf()

reaction(u) = r * u .* (1 .- u)

using LinearAlgebra

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

using DifferentialEquations
prob = ODEProblem(rc_ode, rho0, (0.0, T), saveat=dt)
sol = solve(prob, Tsit5());
ode_data = Array(sol);

figure(figsize=(8,3))

subplot(121)
pcolor(x,t,ode_data')
xlabel("x"); ylabel("t");
colorbar()

subplot(122)
for i in 1:4:Nt
    plot(x, ode_data[:,i], label="t=$(sol.t[i])")
end
legend()
gcf()
tight_layout()
savefig(@sprintf("%s/true_solution.pdf", save_folder))

#### Define neural net for reverse mode AD
using Flux, DiffEqFlux

n_weights = 20
rx_nn = Chain(Dense(1,n_weights,swish),
             Dense(n_weights,2*n_weights,σ),
             Dense(2*n_weights,n_weights,σ),
             Dense(n_weights, n_weights,swish),
             Dense(n_weights,1),
             x -> x[1])
rx_nn_dat = Chain(rx_nn, x -> x.data)

#initialize diffusion coefficient 
D0 = param([0.1])

function nn_ode(u::TrackedArray,p,t)
    du = Tracker.collect(rx_nn.([u[i:i] for i in 1:Nx])) + abs(p[1]) * lap * u
    return du
end

function nn_ode(u::AbstractArray,p,t)
    du = [Flux.data(rx_nn.(u)[i])[1] for i in 1:Nx] + abs(Flux.data(p)[1]) * lap * u
    return du
end

#set up the neural ODE problem
prob_nn = ODEProblem(nn_ode,param(rho0), (0.0, T), D0)
sol_nn = diffeq_rd(D0,prob_nn,Tsit5())

function predict_rd()
  Flux.Tracker.collect(diffeq_rd(D0,prob_nn,Tsit5(),u0=param(rho0),saveat=dt))
end

loss_rd() = sum(abs2, ode_data .- predict_rd())

#Optimization
opt = ADAM(0.01)

using Printf

global count = 0
global save_count = 0
save_freq = 50

train_arr = Float64[]
diff_arr = Float64[]

#set up callback function
cb = function ()
    push!(train_arr, Flux.data(loss_rd()))
    push!(diff_arr, Flux.data(D0)[1])
    println(@sprintf("Loss: %0.4f\t D0: %0.5f", loss_rd(), Flux.data(D0)[1]))

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

        subplot(133)
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
        rx_line.set_ydata(rx_nn_dat.([[elem] for elem in u]))
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
cp("KPP-mixed-rev-mode.jl", @sprintf("%s/runscript.jl", save_folder))
@epochs 500 Flux.train!(loss_rd, params(rx_nn, D0), [()], opt, cb = cb)

#save loss vs epochs plot
figure(figsize=(6,3))
plot(log.(train_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel("Log(loss)")
tight_layout()
savefig(@sprintf("%s/loss_vs_epoch.pdf", save_folder))
gcf()

## Save trained model
using BSON: @save
@save @sprintf("%s/model.bson", save_folder) rx_nn

#save D0 vs epochs plot
figure(figsize=(6,3))
plot(abs.(diff_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel(L"$D$")
ylim([0, 0.02])
axhline(0.010, linestyle="--", color="k", label="Expected Value")
tight_layout()
legend(frameon=false)
savefig(@sprintf("%s/D0_vs_epoch.pdf", save_folder))
gcf()
