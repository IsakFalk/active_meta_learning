using DrWatson
quickactivate("..")

using ActiveMetaLearning
using Distances
using KernelFunctions
using LinearAlgebra
using Statistics
using Random

mse(Y_, Y) = mean((Y_ .- Y).^2)

# Setup Ex1, since r = 0, we have that λ is not used
d = 30
n = 20
n_test = 100
k_test = 20
T_train = 100
T_val = 50
T_test = 200
λ_grid = [10^-6, 10^-3, 10^-1, 10^0, 100]
r_grid = collect(0:3:n-1)
γ = 1.0

function calculate_mean_test_loss(weights::Matrix{<:AbstractFloat}, λ::Float64, test_tasks)
    test_loss = Float64[]
    for i in range(1, stop=size(weights, 1))
        h = mean(weights[1:i, :], dims=1)
        h = reshape(h, :, 1)
        test_loss_ = Float64[]
        for test_task in test_tasks
            X, Y = test_task
            X_tr = X[1:k_test, :]
            Y_tr = reshape(Y[1:k_test], :, 1)
            X_te = X[k_test+1:end, :]
            Y_te = reshape(Y[k_test+1:end], :, 1)
            Cλn = X_tr'*X_tr./k_test + λ.*I
            h_opt = Cλn \ (X_tr'*Y_tr./k_test + λ.*h)
            push!(test_loss_, mse(X_te*h_opt, Y_te))
        end
        push!(test_loss, mean(test_loss_))
    end
    return test_loss
end
function calculate_mean_test_loss(h::Vector{<:AbstractFloat}, λ::Float64, test_tasks)
    test_loss_ = Float64[]
    h = reshape(h, :, 1)
    for test_task in test_tasks
        X, Y = test_task
        X_tr = X[1:k_test, :]
        Y_tr = reshape(Y[1:k_test], :, 1)
        X_te = X[k_test+1:end, :]
        Y_te = reshape(Y[k_test+1:end], :, 1)
        Cλn = X_tr'*X_tr./k_test + λ.*I
        h_opt = Cλn \ (X_tr'*Y_tr./k_test + λ.*h)
        push!(test_loss_, mse(X_te*h_opt, Y_te))
    end
    return mean(test_loss_)
end

# Fix these hyperparameters
h₀ = zeros(d)

function choose_hyperparameters(train_tasks, val_tasks, λ_grid, r_grid)
    println("Finding hyperparameters")
    best_loss = Inf
    opt_params = (0.0, 0)
    for r ∈ r_grid
        for λ ∈ λ_grid
            if r == 0
                weights = learning_to_learn_around_a_common_mean_r0(h₀, γ, train_tasks)
            else
                weights = learning_to_learn_around_a_common_mean(h₀, λ, γ, r, train_tasks)
            end
            h_opt = mean(weights, dims=1)[:]
            current_loss = calculate_mean_test_loss(h_opt, λ, val_tasks)
            if current_loss < best_loss
                println("Current best pair ($(λ), $(r))")
                println("Loss: $(current_loss)")
                best_loss = current_loss
                opt_params = (λ, r)
            end
        end
    end
    return opt_params
end

ex = common_mean_ex1_environment(d)
#ex = common_mean_ex2_environment(d)
runs = 5
test_loss_over_runs = zeros(T_train+1, runs) # train_tasks x runs
test_loss_over_runs_kh = zeros(T_train+1, runs) # train_tasks x runs
Random.seed!(0)

for i in range(1, stop=runs)
    println("Run $(i)")
    train_tasks = get_array_of_tasks(ex, T_train, n)
    val_tasks = get_array_of_tasks(ex, T_val, n_test)
    test_tasks = get_array_of_tasks(ex, T_test, n_test)

    # We only do this for non-KH due to computational reasons
    λ, r = choose_hyperparameters(train_tasks, val_tasks, λ_grid, r_grid)
    println("λ: $(λ), r: $(r), γ: $(γ)")
    # Perform FW to reorder tasks
    ps = CategoricalDistribution[]
    Xs = []
    for task in train_tasks
        X, Y = task
        push!(Xs, X)
        push!(ps, CategoricalDistribution(Matrix(X)))
    end

    # Median trick to set gaussian variance in kernel, twice!
    σ₁ = 1.0
    ρ₁ = 1/(sqrt(2.0)*σ₁) # Represented as exp(-ρ²d(x,y)) not exp(-(1.0/(2σ²)d(x, y))
    mmd = MaximumMeanDiscrepancy(SqExponentialKernel(ρ₁))
    M = mmd(ps)
    σ₂ = median([M[l, s] for l in 2:size(M, 1) for s in 1:l-1])
    ρ₂ = 1.0/(sqrt(2.0 * σ₂))
    k₂_mmd = MMDKernel(SqExponentialKernel(ρ₁), SqExponentialKernel(ρ₂))
    M = kernelmatrix(k₂_mmd, ps, ps)
    kh = EmpiricalDistributionKernelHerding(M)
    order, w = optimize!(kh)
    train_tasks_kh = train_tasks[order]

    # Get loss
    if r == 0
        weights = learning_to_learn_around_a_common_mean_r0(h₀, γ, train_tasks)
        weights_kh = learning_to_learn_around_a_common_mean_r0(h₀, γ, train_tasks_kh)
    else
        weights = learning_to_learn_around_a_common_mean(h₀, λ, γ,  r, train_tasks)
        weights_kh = learning_to_learn_around_a_common_mean(h₀, λ, γ, r, train_tasks_kh)
    end
    test_loss_over_runs[:, i] = calculate_mean_test_loss(weights, λ, test_tasks)
    test_loss_over_runs_kh[:, i] = calculate_mean_test_loss(weights_kh, λ, test_tasks)
end

μ = mean(test_loss_over_runs, dims=2)
σ = std(test_loss_over_runs, dims=2)

μ_kh = mean(test_loss_over_runs_kh, dims=2)
σ_kh = std(test_loss_over_runs_kh, dims=2)

using Plots
pyplot()

plot([μ μ], fillrange=[μ+σ μ-σ], fillalpha=0.1, c=:orange)
plot!([μ_kh μ_kh], fillrange=[μ_kh+σ_kh μ_kh-σ_kh], fillalpha=0.1, c=:blue, title="Blue: KH, Orange: Uniform", xlabel="T", ylabel="Mean MSE over runs (1 CI)")
savefig("example_1_learning_to_learn.jpg")
