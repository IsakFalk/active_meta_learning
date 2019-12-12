using Flux
using Flux: gradient
using LinearAlgebra
using ProgressMeter

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
