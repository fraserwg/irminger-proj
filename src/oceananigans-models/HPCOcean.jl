using SpecialFunctions: erf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface

gaussian(x, L) = exp(-x^2 / 2L^2)

underlying_grid = RectilinearGrid(CPU(), Float32, size = (600, 1000),
                       x = (0, 600kilometers), z = (-1000meters, 0),
                       topology = (Bounded, Flat, Bounded))

h0 = - 180
λ, h1, x_mid, δ = 5.00e-04, 0.5 * (-underlying_grid.Lz + h0), 47500, 3200
bathymetry(x, y) = h0 * (1 - exp(-λ * x)) + h1 * (tanh((x - x_mid) / δ) + 1)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

coriolis = FPlane(f=1.26e-4)


@inline wind_stress(x, y, t, p) = - p.τ * gaussian(t - p.tmid, p.σ_t)



uvel_bcs = FieldBoundaryConditions(immersed = GradientBoundaryCondition(0))

vvel_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress, parameters=(τ=-0.75, tmid=5days, σ_t=0.5days)),
                                   immersed = GradientBoundaryCondition(0),
                                   east = GradientBoundaryCondition(0),
                                   west = GradientBoundaryCondition(0))

model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (:b),
                                    free_surface = ImplicitFreeSurface(),
                                    boundary_conditions = (u=uvel_bcs, v=vvel_bcs,))


Ul = -0.01
Ll = 7kilometers
x₀l = 10kilometers
σzl = -h0

Ur = -0.1
Lr = 10kilometers
x₀r = 65kilometers
σzr = grid.Lz

function vⁱ(x, y, z)
    if z > bathymetry(x, y)
        v = Ur * gaussian(x - x₀r, Lr) .* (z / σzr + 1) + Ul * gaussian(x - x₀l, Ll) .* (z / σzl + 1)
    else
        v = NaN
    end
    return v
end

g = model.free_surface.gravitational_acceleration

function b_ref(z)
    rho_0 = 1000
    z_therm, z_bound = -200 + 1e-9, -500 + 1e-9
    if z > z_therm
        # do the surface profile
        k1, c1 = -0.00060862, 1027.51608543
        rho_ref = k1 * z + c1
    elseif z < z_bound
        # do the deep profile
        k2, c2 = -7.56783898e-05, 1027.68497575
        rho_ref = k2 * z + c2
    else
        a, b, c = -6.922526444855757e-07, -0.0007679310341531434, 1027.511912592355
        # do the intermediate profile
        rho_ref = a * z * z + b * z + c
    end
    
    return - g * rho_ref / rho_0
end

b_from_v(x, xmid, σ_x, σ_z, V0) = - V0 * coriolis.f * σ_x / σ_z * sqrt(π / 2) * erf((xmid - x) / (sqrt(2) * σ_x))

function b_ic(x, y, z)
    if z > bathymetry(x, y)
        b_left = b_from_v.(x, x₀l, Ll, σzl, Ul)
        b_right = b_from_v.(x, x₀r, Lr, σzr, Ur)
        b_ic = + b_left + b_right + b_ref.(z)
    else
        b_ic = NaN
    end

    return b_ic
end

bⁱ(x, y, z) = b_ic(x, y, z)

set!(model, v=vⁱ, η=0, b=bⁱ)


gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
Δt = 0.1wave_propagation_time_scale
Δt = 60

simulation = Simulation(model; Δt, stop_iteration = 1000)

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

output_fields = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = IterationInterval(10),
                                                      filename = "geostrophic_adjustment.jld2",
                                                      overwrite_existing = true)

run!(simulation)

