path = dirname(@__FILE__)
cd(path)


using Turing
using DifferentialEquations

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

using LinearAlgebra

using Optim

# Set a seed for reproducibility.
using Random
Random.seed!(14);


# Define pk model.
function pk_model(du, u, p, t)
    # Model parameters.
    kₐ, kₑ  = p
    A₀, A₁  = u
    # Evaluate differential equations.
    du[1] = - kₐ *  A₀
    du[2] =   kₐ *  A₀ - kₑ * A₁
    return nothing
end

# Define initial-value problem.
u0 = [10.0, 0.0]
p = [0.8, 0.4]
tspan = (0.0, 10.0)
prob = ODEProblem(pk_model, u0, tspan, p)

# Plot simulation.
plot(solve(prob, Tsit5()))

p_dist = MvLogNormal([log(0.8), log(0.4)], [0.3 0.1; 0.1 0.4])

function prob_func(prob, i, repeat)
    remake(prob; p = rand(p_dist))
end
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories = 1000)
sim_summ = EnsembleSummary(sim)
plot(sim_summ, trajectories  = 2)

png("1_plot_mainmodel.png")

ind_param = rand(p_dist)
ind_prob  = ODEProblem(pk_model, u0, tspan, ind_param)
ind_sol   = solve(ind_prob, Tsit5())
plot!(ind_sol, idxs = 2)
png("2_plot_indivmodel.png")

ode_sol   = solve(ind_prob, Tsit5(); saveat=1.0)
ode_ind_sol = getindex.(ode_sol.u, 2)
odedata   = @. rand(LogNormal(log(ode_ind_sol), 0.1))[2:end]
odetime = ode_sol.t[2:end]
scatter!(odetime, odedata)
png("3_plot_indivrnddata.png")


@model function fitlv(data, prob)
    # Prior distributions.
    #σ ~ InverseGamma(2, 3)
    p ~ MvLogNormal([log(0.8), log(0.4)], [0.3 0.1; 0.1 0.4])

    predicted =  getindex.(solve(prob, Rodas5(); p = p, saveat=1.0).u, 2)[2:end]

    # Observations.
    for i in 1:length(predicted)
        if predicted[i] <= 0 predicted[i] = eps() end
        data[i] ~  LogNormal(log(predicted[i]), 0.1)
    end

    return nothing
end

model = fitlv(odedata, prob)
chain = sample(model, NUTS(0.65), MCMCSerial(), 3000, 3; progress=false)
#plot(chain)
#describe(chain)

#plot()
samplen = 500
posterior_samples = Array(sample(chain, samplen; replace=false))

prob_func_post(prob, i, repeat) = begin
    remake(prob; p = posterior_samples[i, :])
end
ensemble_prob_post = EnsembleProblem(prob, prob_func = prob_func_post)
sim_post = solve(ensemble_prob_post, Tsit5(), EnsembleDistributed(), trajectories = samplen)
sim_post_summ = EnsembleSummary(sim_post)
plot!(sim_post_summ, trajectories  = 2)

png("4_plot_indpostmodel.png")

# Plot simulation and noisy observations.
plot()
plot!(solve(prob, Tsit5(); p=[0.8, 0.4]); idxs=(2), linewidth=1, label = "Main model")
plot!(solve(prob, Tsit5(); p=ind_param); idxs=(2), linewidth=1, label = "Main Individual model")
plot!(solve(prob, Tsit5(); p=mean(chain)[:, :mean]); idxs=(2), linewidth=1, label = "Predicted Individual model")
scatter!(odetime, odedata; label = "Individual data")
png("5_plot_indpostmodel_legend.png")

# Несколько доз 


tspan_mult = (0.0, 100.0)
prob_mult = ODEProblem(pk_model, u0, tspan_mult, p)

ensemble_prob_post = EnsembleProblem(prob_mult, prob_func = prob_func_post)

dosetimes = collect(12:12:96)    # Время дозирования
condition(u, t, integrator) = t ∈ dosetimes   # Условие 
affect!(integrator) = integrator.u[1] += 10   # Действие (увеличивает на 10 ед.)
cb = DiscreteCallback(condition, affect!)     # Создает объект для "решателя"



sim_post_mult = solve(ensemble_prob_post, Tsit5(), EnsembleDistributed(), callback = cb, tstops = dosetimes, trajectories = samplen)
sim_post_mult_summ = EnsembleSummary(sim_post_mult)
plot(sim_post_mult_summ, trajectories  = 2)

png("6_plot_ind_mult.png")




# Generate a MLE estimate.
mle_estimate = optimize(model, MLE())

# Generate a MAP estimate.
map_estimate = optimize(model, MAP())

# Import StatsBase to use it's statistical methods.
using StatsBase

# Print out the coefficient table.
coeftable(mle_estimate)