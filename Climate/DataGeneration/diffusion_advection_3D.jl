cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Printf

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters

# Model parameters
N = 128  # Number of grid points in each dimension.
L = 1   # Physical length of each dimension.

κ = 0.05  # Diffusivity
ν = κ     # Assuming Prandtl number Pr = 1

end_time = 10.0

# Boundary conditions
c_bcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Neumann, 0),
                                bottom = BoundaryCondition(Neumann, 0))
model_bcs = HorizontallyPeriodicSolutionBCs(c=c_bcs)

# Set up forcing
@inline Fc(i, j, k, grid, t, U, C, params) = @inbounds cos(sin(C.c[i, j, k]^3)) + sin(cos(C.c[i, j, k]^2))
forcing = ModelForcing(c=Fc)

# Set up model
model = Model(
           architecture = CPU(),
             float_type = Float64,
                   grid = RegularCartesianGrid(size=(N, N, N), x=(0, L), y=(0, L), z=(0, L)),
                tracers = (:c,),
               coriolis = nothing,
               buoyancy = nothing,
                closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
    boundary_conditions = model_bcs,
                forcing = forcing
)

# Setting initial conditions
c₀(x, y, z) = exp(-200(z-0.75)^2)
set!(model, c=c₀)

# Add JLD2 output writer for 3D fields.
fields = Dict(
    :u => model -> Array(interior(model.velocities.u)),
    :v => model -> Array(interior(model.velocities.v)),
    :w => model -> Array(interior(model.velocities.w)),
    :c => model -> Array(interior(model.tracers.c))
)

model.output_writers[:fields] =
    JLD2OutputWriter(model, fields, dir=".", prefix="advection_diffusion_3d_fields",
                     interval=0.1, force=true, verbose=true)

# Add JLD2 output writer for horizontal averages.
 u̅ = HorizontalAverage(model.velocities.u; return_type=Array)
 v̅ = HorizontalAverage(model.velocities.v; return_type=Array)
 w̅ = HorizontalAverage(model.velocities.w; return_type=Array)
 c̅ = HorizontalAverage(model.tracers.c; return_type=Array)

 profiles = Dict(
    :u => model -> u̅(model),
    :v => model -> v̅(model),
    :w => model -> w̅(model),
    :c => model -> c̅(model)
)

model.output_writers[:horizontal_averages] =
    JLD2OutputWriter(model, profiles, dir=".", prefix="advection_diffusion_3d_horizontal_averages",
                     interval=0.01, force=true, verbose=true)

# Set up adaptive time stepping.
wizard = TimeStepWizard(cfl=0.1, Δt=1e-6, max_change=1.2, max_Δt=1e-4)

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
