ENV["GKSwstype"] = 322 # workaround for gr segfault on GH actions
# ENV["GKS_WSTYPE"]=100 # try this if above does not work
using Documenter, IterativeLearningControl, Plots, ControlSystemsBase, LinearAlgebra, JuMP, BlockArrays, OSQP

makedocs(
      sitename = "IterativeLearningControl Documentation",
      doctest = false,
      modules = [IterativeLearningControl],
      warnonly = [:autodocs_block],
      pages = [
            "Home" => "index.md",
            "Algorithms" => [
                  "HeuristicILC" => "HeuristicILC.md",
                  "OptimizationILC" => "OptimizationILC.md",
                  "ConstrainedILC" => "ConstrainedILC.md",
                  "GradientILC" => "GradientILC.md",
                  "ModelFreeILC" => "ModelFreeILC.md",
            ],
            "Manual ILC iterations" => "manual.md",
            "API" => "api.md",
      ],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
)

deploydocs(
      repo = "github.com/baggepinnen/IterativeLearningControl.jl.git",
)
