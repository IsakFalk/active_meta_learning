module ActiveMetaLearning

include("tasks.jl")
include("maml.jl")
include("kernels.jl")

export sample_task, get_array_of_tasks
export MetaLearningEnvironment, MetaLearningRegEnvironment, MetaLearningClassEnvironment
export low_dim_lin_reg_environment, isotropic_lin_reg_environment

export fomaml

export CategoricalDistribution, MaximumMeanDiscrepancy

end
