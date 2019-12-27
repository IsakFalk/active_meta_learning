using KernelFunctions
using Statistics

# Frank-Wolfe has the following problem formulationp
# min J(g)
# s.t g ∈ C
# where J is a twice continuously differentiable objective function
# and C is a compact convex constraint set.
# Frank Wolfe works by calling an oracle
# O(gₜ) = \argmin_{g ∈ C} ⟨J'(gₜ), g⟩
# being able to optimise linear functions on C
abstract type FrankWolfe end

"""
    EmpiricalDistributionKernelFrankWolfe(ρₜ, K)

Run Frank Wolfe on the objective
    ``J(g) = \frac{1}{2}\norm{g - μ}``
s.t ``g ∈ ConvHull({ϕᵢ}_{i=1}^n)``
where ``ϕᵢ = k(x_i, ⋅)`` for some empirical dataset ``{xᵢ}_{i=1}^n``.
"""
struct EmpiricalDistributionKernelFrankWolfe{F, V<:AbstractFloat} <: FrankWolfe
    ρₜ::F
    K::Matrix{V}
    # TODO: Check that K is psdmatrix
end
EmpiricalDistributionKernelHerding(K::Matrix{<:AbstractFloat}) = EmpiricalDistributionKernelFrankWolfe(t -> 1.0 / (1.0 + t), K)

"""
    optimize!(problem)

Return the order and weights of the elements in feature space
gotten by Frank Wolfe.
"""
function optimize!(problem::EmpiricalDistributionKernelFrankWolfe)
    K = problem.K
    ρₜ = problem.ρₜ
    n = size(K, 1)
    # For the first step we just pick the first index
    order = zeros(Int, n)
    w = ones(Float64, n)
    order[1] = 1
    indices_taken = Set(1)
    indices_left = Set(2:n)
    # Calculate the best value and corresponding index for t
    # We only consider the indices left
    for t ∈ range(2, stop=n)
        best_i_so_far = 0
        smallest_value_so_far = Inf
        for i ∈ indices_left
            current_value = 0.0
            current_value -= mean(K[i, :])
            for j ∈ indices_taken
                current_value += w[j] * K[i, j]
            end
            if current_value <= smallest_value_so_far
                smallest_value_so_far = current_value
                best_i_so_far = i
            end
        end
        delete!(indices_left, best_i_so_far)
        order[t] = best_i_so_far
        w[best_i_so_far] *= ρₜ(t-1)
        for j ∈ indices_taken
            w[j] *= (1 - ρₜ(t-1))
        end
        push!(indices_taken, best_i_so_far)
    end
    return order, w[order]
end
