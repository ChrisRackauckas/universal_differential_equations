using Printf

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters

# Model parameters
N = 32  # Number of grid points in each dimension.
L = 1   # Physical length of each dimension.

κ = 0.05  # Diffusivity
ν = κ     # Assuming Prandtl number Pr = 1

end_time = 1

# Set up model
model = Model(
           architecture = CPU(),
             float_type = Float64,
                   grid = RegularCartesianGrid(size=(N, N, N), x=(-L/2, L/2), y=(-L/2, L/2), z=(-L/2, L/2)),
                tracers = (:b,),
               coriolis = nothing,
               buoyancy = BuoyancyTracer(),
                closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
    boundary_conditions = HorizontallyPeriodicSolutionBCs()
)

# Setting initial conditions
ε(σ) = σ * randn()  # Noise
b₀(x, y, z) = 1 * (z < 0) + -1 * (z > 0) + ε(1e-8)
set!(model, b=b₀)

# Add JLD2 output writer for 3D fields.
fields = Dict(
    :u => model -> Array(interior(model.velocities.u)),
    :v => model -> Array(interior(model.velocities.v)),
    :w => model -> Array(interior(model.velocities.w)),
    :b => model -> Array(interior(model.tracers.b))
)

model.output_writers[:fields] =
    JLD2OutputWriter(model, fields, dir=".", prefix="rayleigh_taylor_3d_fields",
                     interval=0.1, force=true, verbose=true)

# Add JLD2 output writer for horizontal averages.
 u̅ = HorizontalAverage(model.velocities.u; return_type=Array)
 v̅ = HorizontalAverage(model.velocities.v; return_type=Array)
 w̅ = HorizontalAverage(model.velocities.w; return_type=Array)
 b̅ = HorizontalAverage(model.tracers.b; return_type=Array)

 profiles = Dict(
    :u => model -> u̅(model),
    :v => model -> v̅(model),
    :w => model -> w̅(model),
    :b => model -> b̅(model)
)

model.output_writers[:horizontal_averages] =
    JLD2OutputWriter(model, profiles, dir=".", prefix="rayleigh_taylor_3d_horizontal_averages",
                     interval=0.01, force=true, verbose=true)

# Set up adaptive time stepping.
wizard = TimeStepWizard(cfl=0.1, Δt=1e-6, max_change=1.2, max_Δt=1e-2)

# Set up CFL diagnostics.
cfl = AdvectiveCFL(wizard)

# Number of time steps to perform at a time before printing a progress
# statement and updating the adaptive time step.
Ni = 20

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
    @printf("[%06.2f%%] i: %d, t: %5.2e days, umax: (%6.3e, %6.3e, %6.3e), CFL: %6.4g, next Δt: %8.5e, ⟨wall time⟩: %s\n",
            progress, i, t, umax, vmax, wmax, cfl(model), wizard.Δt, prettytime(walltime / Ni))
end

