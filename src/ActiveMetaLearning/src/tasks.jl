using Distributions, LinearAlgebra

abstract type MetaLearningEnvironment end
abstract type MetaLearningRegEnvironment <: MetaLearningEnvironment end
abstract type MetaLearningClassEnvironment <: MetaLearningEnvironment end

"""
Low dimensional linear regression task:
Y = X*P*w + E, where Y ∈ R^{n}, P ∈ R^{d × w_d}, X ∈ R^{n × d}, w ∈ R^{w_d}, E ∈ R^{n}
sampled according to p_x, p_w, p_ϵ respectively.
"""
struct low_dim_lin_reg_environment{T} <: MetaLearningRegEnvironment
    # TODO: Put in assertions
    d::Int
    w_d::Int
    P::Matrix{T}
    p_x::Distribution
    p_w::Distribution
    p_ϵ::Distribution
end


function isotropic_lin_reg_environment(d::Int, w_d::Int, P::Matrix{T},
                                       σ_x::Float64=1.0,
                                       σ_w::Float64=1.0,
                                       σ_ϵ::Float64=1.0) where {T<:Float64}
    low_dim_lin_reg_environment(d, w_d, P,
                                MvNormal(d, σ_x),
                                MvNormal(w_d, σ_w),
                                Normal(0.0, σ_ϵ))
end

"""
    sample_task(low_dim_lin_reg_env, n)

Randomly sample a low-dimensional linear regression task ρ and sample
n datapoints (x, y) ∼ ρ iid.

Return X, Y as row-order input and output vectors of type Array{Float64}.
"""
function sample_task(meta_environment::low_dim_lin_reg_environment, n::Int)
    me = meta_environment

    # Sampling from rand when using a d-dimensional r.v. yields
    # a d x n matrix, we transpose this
    X = rand(me.p_x, n)'
    w = rand(me.p_w)
    w = me.P*w
    E = rand(me.p_ϵ, n)
    Y = X*w + E
    return X, Y
end

"""
    get_array_of_tasks(meta_env, T, n)

Compute and return an array of T tasks sampled from `meta_env`, each
task consisting of `n` (x, y) pairs. The array returned is indexed
such that `tasks[t][1][n]` returns the n'th instance from task t and
`tasks[t][2][n]` returns the n'th output.
"""
function get_array_of_tasks(meta_env::MetaLearningEnvironment, T::Int, n::Int)
    tasks = []
    for i in 1:T
        append!(tasks, [sample_task(meta_env, n)])
    end
    return tasks
end
