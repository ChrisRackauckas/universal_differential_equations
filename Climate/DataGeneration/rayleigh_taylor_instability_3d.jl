cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Statistics, Printf
using Plots

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters

# Model parameters
# N = 32  # Number of grid points in each dimension.
Nx = Nz = 128
Ny = 2
L = 1   # Physical length of each dimension.

κ = 1e-4  # Diffusivity
ν = κ     # Assuming Prandtl number Pr = 1

end_time = 10

# Set up model
model = Model(
           architecture = CPU(),
             float_type = Float64,
                   grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-L/2, L/2), y=(-L/2, L/2), z=(-L/2, L/2)),
                tracers = (:b,),
               coriolis = nothing,
               buoyancy = BuoyancyTracer(),
                closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
    boundary_conditions = HorizontallyPeriodicSolutionBCs()
)

# Setting initial conditions
ε(σ) = σ * randn()  # Noise

# b₀(x, y, z) = 1 * (z < 0) + -1 * (z > 0) + ε(1e-2)

@inline interface(x, y, z) = 0.05sin(6π*x)
@inline function b₀(x, y, z)
    z > interface(x, y, z) && return -1
    return 1
end

set!(model, b=b₀)

# Add JLD2 output writer for 3D fields.
fields = Dict(
    :u => model -> Array(interior(model.velocities.u)),
    :v => model -> Array(interior(model.velocities.v)),
    :w => model -> Array(interior(model.velocities.w)),
    :b => model -> Array(interior(model.tracers.b))
)

model.output_writers[:fields] =
    JLD2OutputWriter(model, fields, dir=".", prefix="rayleigh_taylor_instability_3d_fields",
                     interval=1, force=true, verbose=true)

# Add JLD2 output writer for horizontal averages.
 u̅ = HorizontalAverage(model.velocities.u; return_type=Array)
 v̅ = HorizontalAverage(model.velocities.v; return_type=Array)
 w̅ = HorizontalAverage(model.velocities.w; return_type=Array)
 b̅ = HorizontalAverage(model.tracers.b; return_type=Array)

 profiles = Dict(
    :u => model -> u̅(model)[2:end-1],
    :v => model -> v̅(model)[2:end-1],
    :w => model -> w̅(model)[2:end-1],
    :b => model -> b̅(model)[2:end-1]
)

model.output_writers[:horizontal_averages] =
    JLD2OutputWriter(model, profiles, dir=".", prefix="rayleigh_taylor_instability_3d_horizontal_averages",
                     interval=0.1, force=true, verbose=true)

# Set up adaptive time stepping.
wizard = TimeStepWizard(cfl=0.2, Δt=1e-6, max_change=1.2, max_Δt=5e-3)

# Set up CFL diagnostics.
cfl = AdvectiveCFL(wizard)

# Number of time steps to perform at a time before printing a progress
# statement and updating the adaptive time step.
Ni = 20

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

function plot_buoyancy(model)
    t_str = @sprintf("t = %.2f", model.clock.time)
    xC, zC = model.grid.xC, model.grid.zC

    b_horizontal_average = b̅(model)[2:end-1]
    b_profile = plot(b_horizontal_average, zC, title=t_str, label="simulation", xlabel="buoyancy", ylabel="z",
                     xlims=(-1, 1), ylims=(-L/2, L/2))

    coarse_resolution = cr = 16
    b_cr = coarse_grain(b_horizontal_average, cr)
    zC_cr = coarse_grain(collect(zC), cr)

    plot!(b_cr, zC_cr, label="coarsened")

    j½ = Int(model.grid.Ny/2)
    b_xz_data = interior(model.tracers.b)[:, j½, :]'
    b_slice = heatmap(xC, zC, b_xz_data, title="buoyancy slice", color=:balance,
                      xlims=(-L/2, L/2), ylims=(-L/2, L/2), clims=(-1, 1), aspect_ratio=:equal)

    display(plot(b_profile, b_slice, layout=(1, 2), show=true))
end

@time while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)

    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / end_time)

    # Calculate maximum velocities.
    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    # Calculate a new adaptive time step.
    update_Δt!(wizard, model)

    # Print progress statement.
    i, t = model.clock.iteration, model.clock.time
    @printf("[%05.2f%%] i: %d, t: %.2e, umax: (%.3e, %.3e, %.3e), CFL: %.4e, next Δt: %.2e, ⟨wall time⟩: %s\n",
            progress, i, t, umax, vmax, wmax, cfl(model), wizard.Δt, prettytime(walltime / Ni))

    # plot_buoyancy(model)
end

# mp4(anim, "rayleigh_taylor_instability_3d.mp4", fps=15)

@time begin
    # Set up model
    model = Model(
               architecture = CPU(),
                 float_type = Float64,
                       grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(-L/2, L/2), y=(-L/2, L/2), z=(-L/2, L/2)),
                    tracers = (:b,),
                   coriolis = nothing,
                   buoyancy = BuoyancyTracer(),
                    closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
        boundary_conditions = HorizontallyPeriodicSolutionBCs()
    )
    set!(model, b=b₀)
    # Set up adaptive time stepping.
    wizard = TimeStepWizard(cfl=0.2, Δt=1e-6, max_change=1.2, max_Δt=5e-3)

    # Set up CFL diagnostics.
    cfl = AdvectiveCFL(wizard)

    while model.clock.time < end_time
        walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)

        # Calculate simulation progress in %.
        progress = 100 * (model.clock.time / end_time)

        # Calculate maximum velocities.
        umax = maximum(abs, model.velocities.u.data.parent)
        vmax = maximum(abs, model.velocities.v.data.parent)
        wmax = maximum(abs, model.velocities.w.data.parent)

        # Calculate a new adaptive time step.
        update_Δt!(wizard, model)

        # Print progress statement.
        i, t = model.clock.iteration, model.clock.time
        @printf("[%05.2f%%] i: %d, t: %.2e, umax: (%.3e, %.3e, %.3e), CFL: %.4e, next Δt: %.2e, ⟨wall time⟩: %s\n",
                progress, i, t, umax, vmax, wmax, cfl(model), wizard.Δt, prettytime(walltime / Ni))

        # plot_buoyancy(model)
    end
end
