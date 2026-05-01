abstract type AbstractSSM end


"""
    AbstractInvariantLinearGaussianSSM

Represents a linear Gaussian state space model with time-invariant
matrices.
"""
abstract type AbstractInvariantLinearGaussianSSM <: AbstractSSM end

"""
    AbstractLinearGaussianSSM
"""
abstract type AbstractLinearGaussianSSM <: AbstractSSM end

"""
    MomentumCoupledSLSSM <: AbstractInvariantLinearGaussianSSM

A momentum-coupled stochastic logistic state space model.

## Fields

- `modeltype` -- a `String` identifying the type of the model
- `id` -- a `String` identifying the model instance
- `parnames` -- a vector of `String`s supplying names of model parameters
- `Z`
- `T`
- `R`
- `H`
- `Q`
- `a0` -- initial state means
- `P0` -- initial state variances

For the meaning of these, see [`InvariantKalmanRecursion`](@ref).
"""
mutable struct MomentumCoupledSLSSM <: AbstractInvariantLinearGaussianSSM
    modeltype::String
    id::String
    parnames::Vector{String}
    Z::Matrix{Float64}
    T::Matrix{Float64}
    R::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
    a0::Vector{Float64}
    P0::Matrix{Float64}
end


"""
    MomentumCoupledSLSSM(κ:Float64,
                         κp::Float64,
                         σ2y::Float64,
                         σ2z::Float64,
                         σ2s::Float64,
                         qy::Float64,
                         qz::Float64,
                         qs::Float64,
                         a0::Vector{Float64},
                         P0::Matrix{Float64},
                         id::String,
                         parnames::Vector{String})

Constructor for a [`MomentumCoupledSLSSM`](@ref).
"""
function MomentumCoupledSLSSM(κ::Float64,
        κp::Float64,
        σ2y::Float64,
        σ2z::Float64,
        σ2s::Float64,
        qy::Float64,
        qz::Float64,
        qs::Float64,
        a0::Vector{Float64},
        P0::Matrix{Float64},
        id::String,
        parnames::Vector{String})
    Z = zeros(2, 6)
    Z[1,1] = 1.0
    Z[2,4] = 1.0

    #=
    T = [1.0 1.0 0.0 0.0 0.0 0.0;
    0.0 1.0 0.0 κ 0.0 -κ;
    #0.0 1.0 0.0 κ 0.0 0.0;
    1.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 1.0 1.0 0.0;
    κp 0.0 -κp 0.0 1.0 0.0;
    #κp 0.0 0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 1.0 0.0 0.0]
    =#

    T = [1.0 1.0 0.0 κ 0.0 -κ;
         0.0 1.0 0.0 0.0 0.0 0.0;
         1.0 0.0 0.0 0.0 0.0 0.0;
         κp 0.0 -κp 1.0 1.0 0.0;
         0.0 0.0 0.0 0.0 1.0 0.0;
         0.0 0.0 0.0 1.0 0.0 0.0]

    H = [σ2y 0.0;
         0.0 qy * σ2y]

    Q = zeros(6, 6)
    Q[LinearAlgebra.diagind(Q)] = [σ2z, σ2s, 0.0, qz * σ2z, qs * σ2s, 0.0]

    MomentumCoupledSLSSM("MomentumCoupledSLSSM", id, parnames, Z, T, LinearAlgebra.I(6), H, Q, a0, P0)
end


"""
    SLSSM <: AbstractInvariantLinearGaussianSSM

A one-dimensional stochastic logistic state space model.

## Fields

- `modeltype` -- a `String` identifying the type of the model
- `id` -- a `String` identifying the model instance
- `Z`
- `T`
- `R`
- `H`
- `Q`
- `a0` -- initial state means
- `P0` -- initial state variances

For the meaning of these, see [`InvariantKalmanRecursion`](@ref).
"""
mutable struct SLSSM <: AbstractInvariantLinearGaussianSSM
    modeltype::String
    id::String
    parnames::Vector{String}
    Z::Matrix{Float64}
    T::Matrix{Float64}
    R::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
    a0::Vector{Float64}
    P0::Matrix{Float64}
end


mutable struct SSM <: AbstractLinearGaussianSSM
    modeltype::String
    id::String
    parnames::Vector{String}
    Z::Vector{Matrix{Float64}}
    T::Vector{Matrix{Float64}}
    R::Vector{Matrix{Float64}}
    H::Vector{Matrix{Float64}}
    Q::Vector{Matrix{Float64}}
    a0::Vector{Float64}
    P0::Matrix{Float64}
end


"""
    SLSSM(σ2z::Float64,
          σ2s::Float64,
          σ2y::Float64,
          a0::Vector{Float64},
          P0::Matrix{Float64},
          id::String)

Constructor for an [`SLSSM`](@ref).
"""
function SLSSM(σ2z::Float64,
        σ2s::Float64,
        σ2y::Float64,
        a0::Vector{Float64},
        P0::Matrix{Float64},
        id::String)
    Z = [1.0 0.0]
    T = [1.0 1.0; 0.0 1.0]
    R = LinearAlgebra.I(2)
    H = Matrix([σ2y]')
    Q = [σ2z 0.0; 0.0 σ2s]

    return SLSSM("SLSSM", id, Z, T, R, H, Q, a0, P0)
end


"""
    simulate(model::AbstractInvariantLinearGaussianSSM;
             iter::Int,
             dist0)

Fimulate an `AbstractInvariantLinearGaussianSSM` for `iter` time steps
from an initial state sampled from distribution `dist0`.

Returns the simulated series of states, along with simulated series
of observations.
"""
function simulate(model::AbstractInvariantLinearGaussianSSM;
        iter::Int,
        dist0)
    states = []
    obs = []

    distH = Distributions.MvNormal(model.H)
    distQ = Distributions.MvNormal(model.Q)

    push!(states, rand(dist0))

    for t in 1:iter
        newstate = model.T * states[end] + model.R * rand(distQ)
        push!(states, newstate)
        push!(obs, model.Z * newstate + rand(distH))
    end

    (states = states, observations = obs)
end


"""
    fit_restricted_SLSSM(y::Vector{Float64},
              a0,
              P0)

Fit a `model` of type `SLSSM` to data `y`,
assuming that the initial state mean `a0` and variance `P0`.

Employs the BFGS algorithm with automatic differentiation via finite
differences under the hood.
"""
function fit_restricted_SLSSM(y::Vector{Vector{Float64}},
        a0,
        P0)

    function run_model_return_likelihood(u::Vector{Float64}, p)
        mod, kr = run_model_return_model(u, p)

        return -loglikelihood(kr)
    end

    function run_model_return_model(u::Vector{Float64}, p)
        #mod = SLSSM(exp.(u)...)
        mod = SLSSM(0.0, exp(u[1]), exp(u[1]))
        #mod = SLSSM(0.0, exp(u[1]), exp(u[2]) * exp(u[1]))
        kr = kalman_recursion(mod, p.a0, p.P0)
        smooth!(kr, p.data)

        return mod, kr
    end

    success = false
    tries = 0
    sol = nothing

    while !success && tries < 20
        tries += 1
        println(tries)

        try
            u0 = -1 .* rand(1)
            optf = OptimizationFunction(run_model_return_likelihood, ADTypes.AutoFiniteDiff())
            prob = OptimizationProblem(optf, u0, (data = y, a0 = a0, P0 = P0))
            sol = solve(prob, OptimizationOptimJL.BFGS(); maxtime = 30)

            success = SciMLBase.successful_retcode(sol)
        catch e
        end
    end

    println(success)

    if !isnothing(sol)
        mod, kr = run_model_return_model(sol.u, (data = y, a0 = a0, P0 = P0))
        return mod, kr
    else
        return (nothing, nothing)
    end
end


