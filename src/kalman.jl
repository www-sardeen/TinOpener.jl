"""
    AbstractKalmanRecursion

Abstract type that is the supertype of various Kalman recursion structs.
"""
abstract type AbstractKalmanRecursion end


"""
    InvariantKalmanRecursion

A struct to hold the matrices and vectors required for Kalman filtering 
and smoothing in the case the model matrices are time-invariant.
"""
mutable struct InvariantKalmanRecursion <: AbstractKalmanRecursion
    Z::Matrix{Float64}
    T::Matrix{Float64}
    R::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
    a::Vector{Vector{Float64}}
    v::Vector{Vector{Float64}}
    F::Vector{Matrix{Float64}}
    P::Vector{Matrix{Float64}}
    K::Vector{Matrix{Float64}}
    αhat::Vector{Vector{Float64}}
    r::Vector{Vector{Float64}}
    N::Vector{Matrix{Float64}}
    V::Vector{Matrix{Float64}}
end


"""
    kalman_recursion(model::AbstractInvariantLinearGaussianSSM,
                     a::Vector{Float64},
                     P::Matrix{Float64})

Initialize a Kalman recursion for an `AbstractInvariantLinearGaussianSSM`
with initial state mean `a` and variance `P`.
"""
function kalman_recursion(model::AbstractInvariantLinearGaussianSSM,
        a::Vector{Float64},
        P::Matrix{Float64})
    kr = InvariantKalmanRecursion(model.Z,
                                  model.T,
                                  model.R,
                                  model.H,
                                  model.Q,
                                  [],
                                  [],
                                  [],
                                  [],
                                  [],
                                  [],
                                  [],
                                  [],
                                  [])

    push!(kr.a, a)
    push!(kr.P, P)

    push!(kr.r, zeros(size(a)))
    push!(kr.N, zeros(size(kr.T)))

    return kr
end


"""
    forward!(kr::InvariantKalmanRecursion,
            y::Vector{Float64})

Update a Kalman recursion with one data point `y`.
"""
function forward!(kr::InvariantKalmanRecursion, y::Vector{Float64})
    Z = any(isnan.(y)) ? zeros(size(kr.Z)) : kr.Z
    obs = any(isnan.(y)) ? zeros(size(kr.Z * kr.a[end])) : y

    push!(kr.F, Z * kr.P[end] * Z' + kr.H)
    push!(kr.K, kr.T * kr.P[end] * Z' * inv(kr.F[end]))
    push!(kr.v, obs - Z * kr.a[end])
    push!(kr.a, kr.T * kr.a[end] + kr.K[end] * kr.v[end])
    push!(kr.P, kr.T * kr.P[end] * (kr.T - kr.K[end] * Z)' + kr.R * kr.Q * kr.R')
end


"""
    filter!(kr::InvariantKalmanRecursion,
            y::Vector{Vector{Float64}})

Kalman filter.
"""
function filter!(kr::InvariantKalmanRecursion, y::Vector{Vector{Float64}})
    [forward!(kr, z) for z in y]
end


"""
    smooth!(kr::InvariantKalmanRecursion,
            y::Vector{Vector{Float64}})

Kalman smoother.
"""
function smooth!(kr::InvariantKalmanRecursion, y::Vector{Vector{Float64}})
    # Kalman filtering (forward pass)
    filter!(kr, y)

    # Kalman smoothing (backward pass)
    for t in reverse(1:length(kr.K))
        Z = any(isnan.(y[t])) ? zeros(size(kr.Z)) : kr.Z
        L = kr.T - kr.K[t] * Z

        push!(kr.r, Z' * inv(kr.F[t]) * kr.v[t] + L' * kr.r[end])
        push!(kr.N, Z' * inv(kr.F[t]) * Z + L' * kr.N[end] * L)
        push!(kr.αhat, kr.a[t] + kr.P[t] * kr.r[end])
        push!(kr.V, kr.P[t] - kr.P[t] * kr.N[end] * kr.P[t])
    end

    reverse!(kr.r)
    reverse!(kr.N)
    reverse!(kr.αhat)
    reverse!(kr.V)
end


"""
    get_filtered_states(kr::AbstractKalmanRecursion)
    get_filtered_variances(kr::AbstractKalmanRecursion)
    get_smoothed_states(kr::AbstractKalmanRecursion)
    get_smoothed_variances(kr::AbstractKalmanRecursion)

Getters for filtered and smoothed states and variances.
"""
get_filtered_states(kr::AbstractKalmanRecursion) = kr.a[2:(end - 0)]
get_filtered_variances(kr::AbstractKalmanRecursion) = kr.P[2:(end - 0)]
get_smoothed_states(kr::AbstractKalmanRecursion) = kr.αhat
get_smoothed_variances(kr::AbstractKalmanRecursion) = kr.V


"""
    loglikelihood(kr::AbstractKalmanRecursion)

Obtain the log-likelihood from a Kalman filter.
"""
function loglikelihood(kr::AbstractKalmanRecursion)
    n = length(kr.v)
    p = size(kr.v[end])[1]

    return -0.5 * n*p*log(2*pi) - 0.5 * sum([log(LinearAlgebra.det(kr.F[t])) + kr.v[t]' * inv(kr.F[t]) * kr.v[t] for t in 1:n])
end
