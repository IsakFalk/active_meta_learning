using Distributions, LinearAlgebra

abstract type MetaLearningEnvironment end
abstract type MetaLearningRegEnvironment <: MetaLearningEnvironment end
abstract type MetaLearningClassEnvironment <: MetaLearningEnvironment end

function sample_task end

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

########
# MAML #
########

"""
MAML sine wave environment
"""
struct sine_wave_environment <: MetaLearningRegEnvironment
    p_A::Distribution
    p_ϕ::Distribution
    p_x::Distribution
    p_ϵ::Distribution
end
sine_wave_environment() = sine_wave_environment(Uniform(0.1, 5.0), Uniform(0.0, π), Uniform(-5.0, 5.0), DiscreteUniform(0.0, 0.0))

function sample_task(meta_environment::sine_wave_environment, n::Int)
    me = meta_environment
    A = rand(me.p_A)
    ϕ = rand(me.p_ϕ)
    x = rand(me.p_x, n)
    ϵ = rand(me.p_ϵ, n)
    y = A.*sin.(x .+ ϕ) .+ ϵ
    return x, y
end

############################################
# LEO MAML bimodal sine/linear environment #
############################################
struct sine_wave_linear_mixture_environment <: MetaLearningRegEnvironment
    # Mixture probabilities
    p_prior::Bernoulli
    # For sine wave
    p_A::Distribution
    p_ϕ::Distribution
    p_x::Distribution
    # For linear
    p_w::Distribution
    p_b::Distribution
    # Noise
    p_ϵ::Distribution
end
sine_wave_linear_mixture_environment() = sine_wave_linear_mixture_environment(Bernoulli(0.5),
                                                                              Uniform(0.1, 5.0),
                                                                              Uniform(0.0, π),
                                                                              Uniform(-5.0, 5.0),
                                                                              Uniform(-3.0, 3.0),
                                                                              Uniform(-3.0, 3.0),
                                                                              Normal(0.0, 0.3))

function sample_task(meta_environment::sine_wave_linear_mixture_environment, n::Int)
    me = meta_environment
    if rand(me.p_prior) == 0
        A = rand(me.p_A)
        ϕ = rand(me.p_ϕ)
        x = rand(me.p_x, n)
        ϵ = rand(me.p_ϵ, n)
        y = A*sin.(x .+ ϕ) .+ ϵ
    else
        w = rand(me.p_w)
        b = rand(me.p_b)
        x = rand(me.p_x, n)
        ϵ = rand(me.p_ϵ, n)
        y = w*x .+ b .+ ϵ
    end
    return x, y
end


#######################################################
# Learning to Learn around a common mean environments #
#######################################################

"""
Learning to learn around a common mean, Ex. 1
"""
struct common_mean_ex1_environment <: MetaLearningRegEnvironment
    d::Int
end

function sample_task(meta_environment::common_mean_ex1_environment, n::Int)
    d = meta_environment.d
    # Since normal distribution is radial
    # we can sample from it and rescale.
    # This will be the same as sampling uniformly from sphere
    X = randn(n, d)
    X ./= sqrt.(sum(X.^2, dims=2))
    w = randn(d) .+ 4.0
    Y = X*w + (1.0/sqrt(5.0)).*randn(n)
    return X, Y
end

"""
Learning to learn around a common mean, Ex. 2
"""
struct common_mean_ex2_environment <: MetaLearningRegEnvironment
    d::Int
end

function sample_task(meta_environment::common_mean_ex2_environment, n::Int)
    d = meta_environment.d
    w_mog = MixtureModel(
        MvNormal[MvNormal(2.0.*ones(d), 1.0), MvNormal(4.0.*ones(d), 1.0)],
    )
    x_mog = MixtureModel(
        MvNormal[MvNormal(2.0.*ones(d), 1.0), MvNormal(4.0.*ones(d), 1.0)],
    )
    w = reshape(rand(w_mog, 1), :, 1)
    X = rand(x_mog, n)'
    Y = X * w + (1.0/sqrt(5)).*randn(n)
    return X, Y
end

#########################################################################
# Incremental Learning-to-Learn with StatisticalGuarantees environments #
#########################################################################

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
