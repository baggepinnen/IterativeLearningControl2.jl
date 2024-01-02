ENV["GKSwstype"] = 322 # workaround for gr segfault on GH actions
# ENV["GKS_WSTYPE"]=100 # try this if above does not work
using Documenter, IterativeLearningControl2, Plots, ControlSystemsBase, LinearAlgebra, JuMP, BlockArrays, OSQP

makedocs(
      sitename = "IterativeLearningControl2 Documentation",
      doctest = false,
      modules = [IterativeLearningControl2],
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
            "Nonlinear and time-varying systems" => "non_lti.md",
            "API" => "api.md",
      ],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
)

deploydocs(
      repo = "github.com/baggepinnen/IterativeLearningControl2.jl.git",
)
