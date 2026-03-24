"""
    SSMFit

The result of fitting a state space model to data; the output
of [`fit_SSM`](@ref).

## Fields

- `modeltype`: type of model fitted
- `id`: unique identifier for fitted model
- `mod`: fitted model
- `kr`: Kalman recursion of fitted model
- `pars`: model parameters (maximum likelihood estimates)
- `parnames`: names of model parameters
- `npar`: number of model parameters
- `retcode`: return code of model fitting (optimization)
- `ll`: log-likelihood
- `aic`: Akaike information criterion
- `aicc`: Akaike information criterion, small-sample corrected
"""
struct SSMFit
    modeltype::String
    id::String
    mod::AbstractSSM
    kr::AbstractKalmanRecursion
    pars::Vector{Float64}
    parnames::Vector{String}
    npar::Int
    retcode
    ll::Float64
    aic::Float64
    aicc::Float64
end


# pretty-printing SSMFit
#
function Base.show(io::IO, m::MIME"text/plain", x::SSMFit)
    println("SSM fit")
    println("-------")
    println("")
    println("Model type:\t\t$(x.modeltype)")
    println("Model ID:\t\t$(x.id)")
    println("Number of parameters:\t$(x.npar)")
    println("Log-likelihood:\t\t$(x.ll)")
    println("AIC:\t\t\t$(x.aic)")
    println("AICc:\t\t\t$(x.aicc)")
    println("")
    println("Maximum likelihood parameter estimates:")
    for i in 1:length(x.parnames)
        println("\t\t$(x.parnames[i]):\t$(x.pars[i])")
    end
end


# helper function used by fit_SSM
#
function run_model_return_likelihood(u, p)
    mod, kr = run_model_return_model(u, p)
    -loglikelihood(kr)
end


# helper function used by fit_SSM
#
function run_model_return_model(u, p)
    mod = p.fun(u)
    kr = kalman_recursion(mod, mod.a0, mod.P0)
    smooth!(kr, p.data)
    mod, kr
end


"""
    fit_SSM(fun,
            data::Vector{Vector{Float64}},
            npar::Int;
            n_conv = 3,
            max_tries = 100,
            verbose = true)

Fit a state space model to data via maximum likelihood estimation,
using the BFGS algorithm.

## Positional arguments

- `fun`:
- `data`: a vector of vectors of floats; these are the observations in the time series. `NaN` values are allowed as representations of missing data points.
- `npar`: number of model parameters

## Keyword arguments

- `n_conv`: Optimization is repeated from random initial conditions until `n_conv` optimizations have reached convergence
- `max_tries`: Maximum number of optimizations to carry out before exiting, regardless of whether `n_conv` optimizations have converged
- `verbose`: if `true`, prints details to the console as the optimization proceeds

## Value

An [`SSMFit`](@ref) object.
"""
function fit_SSM(fun,
        data::Vector{Vector{Float64}},
        npar::Int;
        n_conv = 3,
        max_tries = 100,
        verbose = true)

    sol = nothing
    successes = 0
    tries = 0
    objective = 10000000000

    if verbose
        p = ProgressMeter.ProgressUnknown(desc = "Fitting state space model...", spinner = true)
    end

    while successes < n_conv && tries < max_tries
        tries += 1
        println(tries)

        try
            u0 = -1 .* rand(npar)

            optf = OptimizationFunction(run_model_return_likelihood, ADTypes.AutoFiniteDiff())
            prob = OptimizationProblem(optf, u0, (fun = fun, data = data))
            sol_here = solve(prob, OptimizationOptimJL.BFGS())

            success = SciMLBase.successful_retcode(sol_here)

            if verbose
                ProgressMeter.next!(p)
            end

            if success
                successes += 1
                if sol_here.objective < objective
                    sol = sol_here
                    objective = sol_here.objective
                end
            end
        catch e
            println(e)
        end
    end

    if verbose
        ProgressMeter.finish!(p)
        println()
    end

    mod, kr = run_model_return_model(sol.u, (fun = fun, data = data))

    pars = sol.u

    N = length(get_smoothed_states(kr))

    SSMFit(mod.modeltype, mod.id, mod, kr, pars, mod.parnames, npar, sol.retcode, loglikelihood(kr), aic(npar, loglikelihood(kr)), aicc(npar, loglikelihood(kr), N))
end
