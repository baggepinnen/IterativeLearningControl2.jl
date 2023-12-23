# Nonlinear and time varying systems

Most algorithms (except [`HeuristicILC`](@ref)) work for nonlinear and/or time varying systems.

## Linear time-varying systems (LTV)
To construct an LTV system model, use the constructor [`LTVSystem`](@ref). LTVSystems can also be obtained by calling [`IterativeLearningControl.linearize`](@ref) on a [`NonlinearSystem`](@ref). LTV models are used in exactly the same way for ILC as LTI models.

```@docs
LTVSystem
```

## Nonlinear systems
To construct a nonlinear system model, use the constructor [`NonlinearSystem`](@ref). To run ILC on a [`NonlinearSystem`](@ref), construct a [`NonlinearILCProblem`](@ref) rather than the standard [`ILCProblem`](@ref).

Everything else behaves the same for nonlinear ILC, i.e., you still call the function [`ilc`](@ref) to run the iterations and the function [`compute_input`](@ref) to compute the ILC input signal manually.

```@docs
NonlinearSystem
NonlinearILCProblem
```

### Linearization
A nonlinear system may be linearized around an operating point or around a trajectory using the function [`IterativeLearningControl.linearize`](@ref). This function returns a `StateSpace` model or an [`LTVSystem`](@ref).

When a [`NonlinearILCProblem`](@ref) is solved, this is performed automatically in the method of [`compute_input`](@ref) associated with the chosen algorithm (if the algorithm is model based).