
#####################################################################
# ЧАСТЬ I
#####################################################################
using DifferentialEquations,  Plots
# Функция возвращает значения производных системы дифф.уравнений
# du - массив куда возвращаются производные
# u - значение функции (массив)
# p - параметры (массив)
# t - время (в данном случае не используется)
function pkmodel_2c2e!(du, u, p, t)   
    kₐ, kₑ, k₁₂, k₂₁, Vmax, Km = p
    A₀, A₁, A₂, A₃, A₄ = u
    du[1] = -kₐ  * A₀
    du[2] =  kₐ  * A₀  -  k₁₂ * A₁  +  k₂₁ * A₂ - kₑ * A₁ - Vmax * A₁ / (Km + A₁)
    du[3] =  k₁₂ * A₁  -  k₂₁ * A₂
    du[4] =  kₑ * A₁
    du[5] = Vmax * A₁  /  (Km + A₁)
end
u0    = [10.000, 0.000, 0.0, 0.0, 0.0]      # Начальные значения
p     = [0.7, 0.1, 0.6, 0.4, 0.15, 0.85]   # Параметры для генерации случайных данных
tspan = (0.0, 48.0)                         # Отрезок времени
prob = ODEProblem(pkmodel_2c2e!, u0, tspan, p) # Формулирование задачи
sol = solve(prob)                              # Решение

# Графики:
p1 = plot(sol,vars=(2))
plot!(collect(0:0.1:48), map(x-> x[4] + x[5], sol(collect(0:0.1:48))), title = "Всего")
plot(plot(sol,vars=(2), title = "Центральный"), 
plot(sol,vars=(4), title = "Выведено"), 
plot(sol,vars=(5), title = "Метаболизировано"), 
p1, layout = 4, legend = false)

using Distributions # Пакет для работы с распределениями (для генерации случайных чисел из нормлаьного распределения)
# Генерация данных
pk_x  = vcat(collect(0.1:0.1:3.9), collect(4:1:24), collect(25:2:48))
sol(pk_x) 
pk_y = map(x->x[2], sol(pk_x)) .* exp.(rand(Normal(0, 0.05), length(pk_x)))
scatter(pk_x, pk_y, legend = false)


using  LsqFit                                                 # Пакет для подгонки нелинейных моедлей
function pkmodel_2c2e_vec(time_array, param)                  # Векторная функция R -> Rn
    tspan = (0, time_array[end])                              # Отрезок времени
    u0    = [10.000, 0.000, 0.0, 0.0, 0.0]                    # Начальные условия
    oprob = ODEProblem(pkmodel_2c2e!, u0, tspan, exp.(param)) # Задача
    osol  = solve(oprob, Tsit5(), saveat = time_array)        # Решение
    return @. log(getindex(osol.u, 2))                        # Вывод
end
p0 = log.([0.5, 0.2, 0.4, 0.3, 0.2, 1.65])                # начальные параметры подгонки
fit = curve_fit(pkmodel_2c2e_vec, pk_x, log.(pk_y), p0)   # подгонка


# Параметры                             
param = exp.(fit.param) 
# Решение для найденных параметров
prob = ODEProblem(pkmodel_2c2e!, u0, tspan, param) 
sol = solve(prob)                        
plot!(sol,vars=(2))

println("Параметры:")    
display(param)

println("Доверительные интервалы:")
display(map(x-> exp.(x), confidence_interval(fit, 0.1)))


#####################################################################
# ЧАСТЬ II
#####################################################################

# ФК Модель, eq 3-5
function pkev(du, u, p, t) # Функция
    kₐ, kₑ, k₁₂,k₂₁ = p
    A, Xc, Xp = u
    du[1] = - kₐ * A
    du[2] = kₐ * A - k₁₂ * Xc - kₑ * Xc + k₂₁ * Xp
    du[3] = k₁₂ * Xc - k₂₁ * Xp
end
pk_u0 = [10.0, 0.0, 0.0]     # Начальные условия
pk_p = [0.4, 0.2, 0.1, 0.04] # Параметры
tspan = (0.0, 180.0)         # Отрезок времени
# Задача
pkprob = ODEProblem(pkev, pk_u0, tspan, pk_p)
# Решение
pkevsol     = solve(pkprob)
pk_sol_f(x) = pkevsol(x)[3]
plot(plot(pkevsol; label = ["A" "C" "P"], xlims = (0, 60)), plot(pk_sol_f, xlims = (0, 180)))

using Sundials               # Дополнительные решатели нелинейных ДУ 
# ФД Модель 4, eq 8 - 12
function pd(du, u, p, t)
    L₀, L₁, ω, κ₁, κ₂ = p
    C = pk_sol_f(t)             # Концентрация от времени
    x₁, x₂, x₃, x₄ = u
    ω = x₁ + x₂ + x₃ + x₄
    du[1] = (2L₀*L₁*x₁^2)/(L₁+2L₀*x₁)/ω - κ₂ * C * x₁
    du[2] = κ₂ * C * x₁ - κ₁ * x₂
    du[3] = κ₁ * x₂ - κ₁ * x₃
    du[4] = κ₁ * x₃ - κ₁ * x₄
end
ω₀ = 10.0
pd_u0 = [ω₀, 0.0, 0.0, 0.0] # Начальные условия
pd_p = [0.5, 2., 25.0, 0.05, 0.045] # Параметры
    # Задача
pdprob = ODEProblem(pd, pd_u0, tspan, pd_p)
    # Решение
sol = solve(pdprob, CVODE_BDF())
    # График
plot(sol; label = ["X1" "X2" "X3" "X4"])



# Несколько доз 
dosetimes = [48.0, 94.0]                      # Время дозирования
condition(u, t, integrator) = t ∈ dosetimes   # Условие 
affect!(integrator) = integrator.u[1] += 10   # Действие (увеличивает на 10 ед.)
cb = DiscreteCallback(condition, affect!)     # Создает объект для "решателя"
pkmsol = solve(pkprob, Tsit5(), callback = cb, tstops = dosetimes) #  Решение

pk_sol_mult_f(x) = pkmsol(x)[3]  # Новая ФК функция с многократным дозированием
plot(pk_sol_mult_f, xlims = (0, 180))

# ФД модель с многократным дозированием
function pd_mult(du, u, p, t)
    L₀, L₁, ω, κ₁, κ₂ = p
    C = pk_sol_mult_f(t)             # Концентрация от времени
    x₁, x₂, x₃, x₄ = u
    ω = x₁ + x₂ + x₃ + x₄
    du[1] = (2L₀*L₁*x₁^2)/(L₁+2L₀*x₁)/ω - κ₂ * C * x₁
    du[2] = κ₂ * C * x₁ - κ₁ * x₂
    du[3] = κ₁ * x₂ - κ₁ * x₃
    du[4] = κ₁ * x₃ - κ₁ * x₄
end
ω₀ = 10.0
pd_u0 = [ω₀, 0.0, 0.0, 0.0] # Начальные условия
pd_p = [0.5, 2., 25.0, 0.05, 0.045] # Параметры
    # Задача
pdprob = ODEProblem(pd_mult, pd_u0, tspan, pd_p)
    # Решение
pd_mult_sol = solve(pdprob, CVODE_BDF())

plot(pk_sol_mult_f, label = "PK", xlims = (0, 180), legend=:topleft)
plot!(twinx(), pd_mult_sol, vars=(1), label = "PD",color=:red)


#####################################################################
# ЧАСТЬ III
#####################################################################

function σ_pkev(du, u, p, t)
    du[1] = u[1]*0.2
    du[2] = 0.0
    du[3] = 0.0
end
prob_sde_pkev = SDEProblem(pkev, σ_pkev, pk_u0, tspan, pk_p)
sde_sol = solve(prob_sde_pkev, callback = cb, tstops = dosetimes)
plot(sde_sol, xlims = (0, 180))

# Ансамбли
ensembleprob_pk = EnsembleProblem(prob_sde_pkev)
ens_pk_sol = solve(ensembleprob_pk, EnsembleThreads(), trajectories = 1000, callback = cb, tstops = dosetimes)
summ_pk = EnsembleSummary(ens_pk_sol, 0:0.1:180)
plot(summ_pk, xlims = (0, 180))


# Полная ФК-ФД модель
function pd_sde(du, u, p, t)
    L₀, L₁, ω, κ₁, κ₂ = p[5:9]
    x₁, x₂, x₃, x₄ = u[4:7]
    ω = x₁ + x₂ + x₃ + x₄
    kₐ, kₑ, k₁₂,k₂₁ = p[1:4]
    A, Xc, Xp = u[1:3]
    if kₐ < 0 kₐ = 0 end
    C = u[3]
    if u[4] < 0.01 u[4] = 0.0 end
    du[1] = - kₐ * A
    du[2] = kₐ * A - k₁₂ * Xc - kₑ * Xc + k₂₁ * Xp
    du[3] = k₁₂ * Xc - k₂₁ * Xp
    du[4] = (2L₀*L₁*x₁^2)/(L₁+2L₀*x₁)/ω - κ₂ * C * x₁
    du[5] = κ₂ * C * x₁ - κ₁ * x₂
    du[6] = κ₁ * x₂ - κ₁ * x₃
    du[7] = κ₁ * x₃ - κ₁ * x₄
end
ω₀ = 10.0
pkpd_u0 = [10.0, 0.0, 0.0, ω₀, 0.0, 0.0, 0.0]              # Начальные условия
pkpd_p = [0.4, 0.2, 0.1, 0.04, 0.5, 2., 25.0, 0.05, 0.045] # Параметры
   
function σ_pkpdev(du, u, p, t)
    du[1] = u[1]*0.4
    du[2] = 0.0
    du[3] = 0.0
    du[4] = 0.0
    du[5] = 0.0
    du[6] = 0.0
    du[7] = 0.0
end

prob_sde_pkpd     = SDEProblem(pd_sde, σ_pkpdev, pkpd_u0, tspan, pkpd_p)
ensembleprob_pkpd = EnsembleProblem(prob_sde_pkpd)

ens_pkpd_sol = solve(ensembleprob_pkpd, EnsembleThreads(), trajectories = 1000, callback = cb, tstops = dosetimes)
summ_pkpd = EnsembleSummary(ens_pkpd_sol, 0:0.1:180)
plot(summ_pkpd,   xlims = (0, 180), trajectories  = (4, 3))

#=
# График NLME вывода
using DataFrames, CSV
path = dirname(@__FILE__)
cd(path)
nlmedf = CSV.File("inspobj.csv") |> DataFrame
scatter(nlmedf.dv, nlmedf.dv_ipred, ylabel = "pred",  xlabel = "obs")
=#