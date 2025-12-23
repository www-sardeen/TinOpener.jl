"""
    AbstractInvariantLinearGaussianSSM

Represents a linear Gaussian state space model with time-invariant
matrices.
"""
abstract type AbstractInvariantLinearGaussianSSM end


"""
    MomentumCoupledSLSSM

A momentum-coupled stochastic logistic state space model.
"""
mutable struct MomentumCoupledSLSSM <: AbstractInvariantLinearGaussianSSM
    Z::Matrix{Float64}
    T::Matrix{Float64}
    R::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
end

function MomentumCoupledSLSSM(κ::Float64,
        κp::Float64,
        σ2y::Float64,
        σ2z::Float64,
        σ2s::Float64,
        qy::Float64,
        qz::Float64,
        qs::Float64)
    Z = zeros(2, 6)
    Z[1,1] = 1.0
    Z[2,4] = 1.0

    T = [1.0 1.0 0.0 0.0 0.0 0.0;
         0.0 1.0 0.0 κ 0.0 -κ;
         1.0 0.0 0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 1.0 1.0 0.0;
         κp 0.0 -κp 0.0 1.0 0.0;
         0.0 0.0 0.0 1.0 0.0 0.0]

    H = [σ2y 0.0;
         0.0 qy * σ2y]

    Q = zeros(6, 6)
    Q[LinearAlgebra.diagind(Q)] = [σ2z, σ2s, 0.0, qz * σ2z, qs * σ2s, 0.0]

    MomentumCoupledSLSSM(Z, T, LinearAlgebra.I(6), H, Q)
end


"""
    SLSSM

A one-dimensional stochastic logistic state space model.
"""
mutable struct SLSSM <: AbstractInvariantLinearGaussianSSM
    Z::Matrix{Float64}
    T::Matrix{Float64}
    R::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
end

