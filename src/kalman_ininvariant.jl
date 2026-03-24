"""
    KalmanRecursion <: AbstractKalmanRecursion

A struct to hold the matrices and vectors required for Kalman filtering 
and smoothing in the case the model matrices are time-invariant.

## Fields

For the meaning of the fields of this struct, consult
Durbin & Koopman (2012), particularly the filtering and smoothing
recursion equations (4.24) and (4.44) and their surrounding text.

- `Z` -- the observation matrix
- `T` -- the process matrix
- `R` -- process noise matrix
- `H` -- covariances of observation noise
- `Q` -- covariances of process noise
- `a` -- state means
- `v` -- difference between observation and expected observation (i.e. forecast error, i.e. "innovation")
- `F` -- variance of `v`
- `P` -- state variances
- `K` -- Kalman gain
- `αhat` -- smoothed state mean
- `r` -- weighted sum of innovations, used in the computation of `αhat`
- `N` -- used in the computation of `V`
- `V` -- smoothed state variance
- `t` -- current time

## Reference

Durbin, J. & Koopman, S. J. (2012) Time series analysis by state
space methods. 2nd edition. Oxford: Oxford University Press.
"""
mutable struct KalmanRecursion <: AbstractKalmanRecursion
    Z::Vector{Matrix{Float64}}
    T::Vector{Matrix{Float64}}
    R::Vector{Matrix{Float64}}
    H::Vector{Matrix{Float64}}
    Q::Vector{Matrix{Float64}}
    a::Vector{Vector{Float64}}
    v::Vector{Vector{Float64}}
    F::Vector{Matrix{Float64}}
    P::Vector{Matrix{Float64}}
    K::Vector{Matrix{Float64}}
    αhat::Vector{Vector{Float64}}
    r::Vector{Vector{Float64}}
    N::Vector{Matrix{Float64}}
    V::Vector{Matrix{Float64}}
    t::Int
end


"""
    kalman_recursion(model::AbstractLinearGaussianSSM,
                     a::Vector{Float64},
                     P::Matrix{Float64})

Initialize a Kalman recursion for any descendant of
[`AbstractLinearGaussianSSM`](@ref),
with (known/assumed) initial state mean `a` and variance `P`.

## Value

A [KalmanRecursion](@ref) object.
"""
function kalman_recursion(model::AbstractLinearGaussianSSM,
        a::Vector{Float64},
        P::Matrix{Float64})
    kr = KalmanRecursion(model.Z,
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
                                  [],
                                 0)

    push!(kr.a, a)
    push!(kr.P, P)

    push!(kr.r, zeros(size(a)))
    push!(kr.N, zeros(size(kr.T[1])))

    return kr
end


"""
    forward!(kr::KalmanRecursion,
             y::Vector{Float64})

Update a Kalman recursion with one data point `y`.
"""
function forward!(kr::KalmanRecursion, y::Vector{Float64})
    kr.t += 1
    t = kr.t

    Z = any(isnan.(y)) ? zeros(size(kr.Z[t])) : kr.Z[t]
    obs = any(isnan.(y)) ? zeros(size(Z * kr.a[end])) : y

    push!(kr.F, Z * kr.P[end] * (Z)' + kr.H[t])
    push!(kr.K, kr.T[t] * kr.P[end] * (Z)' * inv(kr.F[end]))
    push!(kr.v, obs - Z * kr.a[end])
    push!(kr.a, kr.T[t] * kr.a[end] + kr.K[end] * kr.v[end])
    push!(kr.P, kr.T[t] * kr.P[end] * (kr.T[t] - kr.K[end] * Z)' + kr.R[t] * kr.Q[t] * (kr.R[t])')
end


"""
    filter!(kr::KalmanRecursion,
            y::Vector{Vector{Float64}})

Kalman filter.
"""
function filter!(kr::KalmanRecursion, y::Vector{Vector{Float64}})
    [forward!(kr, z) for z in y]
end


"""
    smooth!(kr::KalmanRecursion,
            y::Vector{Vector{Float64}})

Kalman smoother.
"""
function smooth!(kr::KalmanRecursion, y::Vector{Vector{Float64}})
    # Kalman filtering (forward pass)
    filter!(kr, y)

    # Kalman smoothing (backward pass)
    for t in reverse(1:length(kr.K))
        Z = any(isnan.(y[t])) ? zeros(size(kr.Z[t])) : kr.Z[t]
        L = kr.T[t] - kr.K[t] * Z

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


