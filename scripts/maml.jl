"""
MAML experiments including FOMAML and variants
"""
using ActiveMetaLearning
using Flux

############
# (FO)MAML #
############

# Create regression task
d = 5
w_d = 2
P = randn(d, w_d)
meta_env = isotropic_lin_reg_environment(d, w_d, P)

# Sample an array of tasks beforehand
T = 100
n = 50
tasks = get_array_of_tasks(meta_env, T, n)
test_tasks = get_array_of_tasks(meta_env, n, T)

### FOMAML hyper-params
# K-shot
K = 5
# α, β: step-size for inner and outer loop respectively
# and number of gd updates (non-stochastic due to MMD bound)
# requiring deterministic and smooth algorithms
α = 0.1
β = 0.1

# Model
h = 100
model = Chain(
    Dense(d, h, Flux.σ; initW=zeros, initb=zeros),
    Dense(h, 1, Flux.identity; initW=zeros, initb=zeros)
)

correct_model = Chain(
    Dense(d, 1, Flux.identity; initW=zeros, initb=zeros)
)

test_loss = fomaml(correct_model, tasks, test_tasks, K; meta_opt=Descent(β), inner_opt=Descent(α))

test_loss_ = Float64[]
for test_task in test_tasks
    X, Y = test_task
    X_tr, X_val = X[1:K, :], X[K+1:end, :]
    Y_tr, Y_val = reshape(Y[1:K], :, 1), reshape(Y[K+1:end], :, 1)
    # Update base model, evaluate on validation meta-set
    push!(test_loss_, Flux.mse(X_val'*0, Y_val'))
end
mean(test_loss_)

# Save results
