cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DelimitedFiles, Plots
# Training takes a while, so save results of simulations and
# have this separate plot script so can tweak plots without
#  rerunning simulations

# or whatever path to data
er_outfile = "er_data.txt"
plt_outfile = "plt_data.txt"

er_dat = readdlm(er_outfile, '\t', Float32, '\n')
plt_dat = readdlm(plt_outfile, '\t', Float32, '\n')

p1 = plot(er_dat[:,1],xscale=:log10,yscale=:log10,ylabel="Error",xlabel="Training steps",
        label="Training error, Neural net")
plot!(p1,er_dat[:,2],label="Testing error, Neural net")
plot!(p1,er_dat[:,3], label="Training error, linear model")
plot!(p1,er_dat[:,4],label="Testing error, linear model")

p2 = plot(plt_dat[:,1],plt_dat[:,3],m=:hexagon, lw=3, ms=3,
        label="Linear model",ylabel="stress",xlabel="time")
plot!(p2,plt_dat[:,1],plt_dat[:,2],m=:circle,lw=3,ms=3,
        label="NN solution")
plot!(p2,plt_dat[:,1],plt_dat[:,4],label="True solution",lw=3,lc=:black,
        xlims=(0,7),leg=:bottomright)

pt = plot(p1,p2,layout=(2,1),size=(600,500))

# or path to wherever
savefig(pt, "CombFig.pdf")
