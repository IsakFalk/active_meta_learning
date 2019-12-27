using Flux
using Flux: gradient
using LinearAlgebra
using ProgressMeter
import MultivariateStats

mse(Y_, Y) = mean((Y_ .- Y).^2)

########
# MAML #
########

"""
    base_model_update(model, X_tr, Y_tr)

Update the model by doing one GD step with respect to the mse of
model(X_tr') and Y_tr', where X_tr, Y_tr is row-major form.

Return the new model and parameters fine-tuned on the task.
"""
function base_model_update(model, opt, X_tr, Y_tr)
    # Create new parameters to perform one-step update
    new_model = deepcopy(model)
    θ_new = Flux.params(new_model)
    # Inner update
    gs = gradient(() -> Flux.mse(new_model(X_tr'), Y_tr'), θ_new)
    Flux.Optimise.update!(opt, θ_new, gs)
    return new_model, θ_new
end

"""
    fomaml(model, tasks, test_tasks, K[, meta_op, inner_opt])

Run fomaml using model on the sequence of tasks in tasks, aggregating
the test loss after each iteration by evaluating the current meta algorithm
on test_tasks. Update the models using meta_opt and inner_opt respectively
doing K-shot supervised learning.

Note: This is run sequentially and is not the actual algorithm as specified
in the maml paper.
"""
function fomaml(model, tasks::Array, test_tasks::Array, K=10; meta_opt=Descent(0.01), inner_opt=Descent(0.02))
    test_loss = Float64[]
    for (t, task) in enumerate(tasks)
        if t % 10 == 0
            println("At iteration $(t)")
        end

        X, Y = task
        X_tr, X_val = X[1:K, :], X[K+1:end, :]
        Y_tr, Y_val = reshape(Y[1:K], :, 1), reshape(Y[K+1:end], :, 1)

        # Update base model
        new_model, θ_new = base_model_update(model, inner_opt, X_tr, Y_tr)
        # Update meta model
        gs = gradient(() -> Flux.mse(new_model(X_val'), Y_val'), θ_new)
        Flux.Optimise.update!(meta_opt, θ_new, gs)
        model = new_model

        test_loss_ = Float64[]
        for test_task in test_tasks
            X, Y = test_task
            X_tr, X_val = X[1:K, :], X[K+1:end, :]
            Y_tr, Y_val = reshape(Y[1:K], :, 1), reshape(Y[K+1:end], :, 1)
            # Update base model, evaluate on validation meta-set
            new_model, _ = base_model_update(model, inner_opt, X_tr, Y_tr)
            push!(test_loss_, Flux.mse(new_model(X_val'), Y_val'))
        end
        push!(test_loss, mean(test_loss_))
    end
    return test_loss
end

##########################################
# Learning to learn around a common mean #
##########################################
function learning_to_learn_around_a_common_mean(h₀::Vector{<:AbstractFloat}, λ::V, γ::V, r::Int, tasks::Array) where {V<:AbstractFloat}
    # Save the weights of each t here
    T = length(tasks)
    X, _ = tasks[1]
    n, d = size(X)
    @assert n > r "r=$(r) has to be strictly smaller than n=$(n)"
    @assert d == length(h₀) "the weight dimension $(size(h₀)) not the same as input dimension $(d)"
    weights = zeros(eltype(X), T+1, d)
    weights[1, :] = h₀

    take_gd_step(h, X, Y) = h - γ.*X'*(X*h - Y)
    function build_meta_regression_data(Xᵗ, Yᵗ)
        # Split data into (n-r), (r) matrix
        Xᵗr = Xᵗ[1:r, :]
        Xᵗnmr = Xᵗ[r+1:end, :]
        Yᵗr = reshape(Yᵗ[1:r], :, 1)
        Yᵗnmr = reshape(Yᵗ[r+1:end], :, 1)
        Cλr = (Xᵗr'*Xᵗr./n) + λ.*I
        # Prefix b for `bar`
        bXᵗr = (Cλr \ (Xᵗnmr'))'.*(λ/sqrt(n - r))
        bYᵗr = (1.0/sqrt(n - r)).*(Yᵗnmr - (sqrt(n - r)/λ).*bXᵗr*(Xᵗr'*Yᵗr./n))
        return bXᵗr, bYᵗr
    end
    for (t, task) in enumerate(tasks)
        Xᵗn, Yᵗn = task
        bXᵗr, bYᵗr = build_meta_regression_data(Xᵗn, Yᵗn)
        h_prev = reshape(weights[t, :], :, 1)
        weights[t+1, :] = take_gd_step(h_prev, bXᵗr, bYᵗr)
    end
    return weights
end

function learning_to_learn_around_a_common_mean_r0(h₀::Vector{<:AbstractFloat}, γ::V, tasks::Array) where {V<:AbstractFloat}
    # Save the weights of each t here
    T = length(tasks)
    X, _ = tasks[1]
    n, d = size(X)
    @assert d == length(h₀) "the weight dimension $(size(h₀)) not the same as input dimension $(d)"
    weights = zeros(eltype(X), T+1, d)
    weights[1, :] = h₀

    take_gd_step(h, X, Y) = h - (γ/n).*X'*(X*h - Y)
    for (t, task) in enumerate(tasks)
        Xᵗ, Yᵗ = task
        h_prev = reshape(weights[t, :], :, 1)
        weights[t+1, :] = take_gd_step(h_prev, Xᵗ, Yᵗ)
    end
    return weights
end

# Wrong need to fix, look at paper
function independent_task_learning(λ::T, tasks::Array, test_tasks::Array) where {T<:AbstractFloat}
    function calculate_mean_test_loss(h)
        test_loss_ = Float64[]
        for test_task in test_tasks
            X, Y = test_task
            # Update base model, evaluate on validation meta-set)
            push!(test_loss_, mse(X*h_output))
        end
        return mean(test_loss_)
    end
    Xs = []
    Ys = []
    for task in tasks
        X, Y = task
        push!(Xs, X)
        push!(Ys, Y)
    end
    Xs = vcat(Xs...)
    Ys = vcat(Ys...)
    h_ridge = MultivariateStats.ridge(Xs', Ys', λ)
    h_ridge = reshape(h_ridge, :, 1)
    for (t, task) in enumerate(tasks)
        if t % 10 == 0
            println("At iteration $(t)")
        end
        Xᵗ, Yᵗ = task
        h_prev = reshape(weight[t, :], :, 1)
        weight[t+1, :] = take_gd_step(h_prev, Xᵗ, Yᵗ)
        push!(test_loss, calculate_mean_test_loss(ridge_reg_h))
    end
    return test_loss
end
