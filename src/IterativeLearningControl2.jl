module IterativeLearningControl2
using ControlSystemsBase, RecipesBase, LinearAlgebra, Statistics

export ilc,
    OptimizationILC, HeuristicILC, ConstrainedILC, ModelFreeILC, GradientILC,
    ILCProblem,
    init, compute_input,
    ilc_theorem


function lsim_zerophase(G, u, args...; kwargs...)
    res = lsim(G, u[:, end:-1:1], args...; kwargs...)
    lsim(G, res.y[:, end:-1:1], args...; kwargs...).y
end

function lsim_noncausal(L::LTISystem{<:Discrete}, u, args...; kwargs...)
    L isa AbstractStateSpace && (return lsim(L, u, args...; kwargs...))
    np = length(denpoly(L)[])
    nz = length(numpoly(L)[])
    zeroexcess = nz-np
    if zeroexcess <= 0
        return lsim(L, u, args...; kwargs...)
    end
    integrators = tf(1, [1, 0], L.Ts)^zeroexcess
    res = lsim(L*integrators, u, args...; kwargs...)
    res.y[1:end-zeroexcess] .= res.y[1+zeroexcess:end]
    res.y
end

function hankel_operator(x,L::Int)
    H = zeros(L, L)
    for i = 1:L
        H[i:end, i] = x[1:end-i]
    end
    H
end

nadjustment(sys::LTISystem) = ControlSystemsBase.ninputs(sys)

"""
    hankel(sys::LTISystem{<:Discrete}, N::Int)

Return a matrix operator ``H`` such that ``Hu^T = y^T`` where `y = lsim(sys, u)`. ``H`` is a Hankel matrix containing the Markov parameters of the system (scaled impulse response).
"""
function hankel_operator(sys::LTISystem{<:Discrete}, N::Int)
    ControlSystemsBase.issiso(sys) || return mv_hankel_operator(sys, N)
    Tf = N*sys.Ts
    imp = impulse(sys, Tf).y[:] .* sys.Ts
    hankel_operator(imp, N)
end

"""
    ILCSolution

A structure representing the solution to an ILC problem. 

# Fields:
- `Y`: Plant responses. `Y[i]` is the response during the `i`th iteration. The first response is recorded before any ILC input is applied, while the last response is recorded *before* the last ILC adjustment signal has been computed, i.e., `Y[end]` corresponds to the response to the ILC input `A[end-1]`.
- `E`: Errors. `E[i]` is the error during the `i`th iteration, i.e., `E[i] = r - Y[i]` where `r` is the reference signal.
- `A`: ILC inputs. `A[i]` is the ILC input during the `i`th iteration.
- `prob`: The `ILCProblem` that was solved
- `alg`: The `ILCAlgorithm` that was used
"""
struct ILCSolution
    Y
    E
    A
    prob
    alg
end

"""
    ILCProblem

# Fields:
- `r`: Reference signal, a matrix of size `(ny, N)` where `ny` is the number of outputs and `N` is the number of time points.
- `Gr`: Closed-loop transfer function from reference to output. May be either an `LTISystem` model from ControlSystemsBase, such as those created using `tf` or `ss`, or an [`LTVSystem`](@ref) model.
- `Gu`: Closed-loop transfer function from plant input to output
"""
struct ILCProblem
    r
    Gr
    Gu
end
nadjustment(prob::ILCProblem) = nadjustment(prob.Gu)


"""
    ILCProblem(; r, Gr, Gu)
    ILCProblem(; r, P, C)

Construct an ILCProblem given discrete-time transfer function models of either
- The closed-loop transfer functions from reference to output and from plant input to output, or
- The plant and controller transfer functions

Continuous-time transfer functions can be discretized using the function `ControlSystemsBase.c2d`.

- `r`: Reference trajectory
- `Gr`: Closed-loop transfer function from reference to output
- `Gu`: Closed-loop transfer function from plant input to output
- `P`: Plant model
- `C`: Controller transfer function
"""
function ILCProblem(; r, Gr=nothing, Gu=nothing, P=nothing, C=nothing)
    if Gr !== nothing && Gu !== nothing

    elseif P !== nothing && C !== nothing
        Gr = feedback(P*C)
        Gu = feedback(P, C)
    else
        error("Either (Gr, Gu) or (P, C) must be provided")
    end
    isdiscrete(Gr) && isdiscrete(Gu) || error("Gr and Gu must be discrete-time transfer functions. Continuous-time transfer function can be discretized using the function c2d.")
    ILCProblem(r, Gr, Gu)
end

simulate(prob, alg, a) = lsim([prob.Gr prob.Gu], [prob.r; a])

abstract type ILCAlgorithm end

"""
    workspace = init(prob, alg)

Initialize the ILC algorithm. This function is called internally by the funciton [`ilc`](@ref) but manual iterations require the user to initialize the workspace explicitly.
"""
init(prob, alg) = nothing

"""
    HeuristicILC(  Q, L, location)        # Positional arguments
    HeuristicILC(; Q, L, location = :ref) # Keyword arguments

Apply the learning rule

```math
\\begin{aligned}
y_k(t) &= G_r(q) \\big(r(t) + a_k(t) \\big) \\\\
e_k(t) &= r(t) - y_k(t) \\\\
a_k(t) &= Q(q) \\big( a_{k-1}(t) + L(q) e_{k-1}(t) \\big)
\\end{aligned}
```

If `location = :input`, the first equation above is replaced by
```math
y_k(t) = G_r(q) r(t) + G_u(q) a_k(t)
```

A theorem due to Norrlöf says that for this ILC iterations to converge, one needs to satisfy
```math
| 1 - LG | < |Q^{-1}|
```
which we can verify by looking at the plot produced by the [`ilc_theorem`](@ref) function.

# Fields:
- `Q(z)`: Robustness filter (discrete time). The filter will be applied both forwards and backwards in time (like `filtfilt`), and the effective filter transfer funciton is thus ``Q(z)Q(z̄)``.
- `L(z)`: Learning filter (discrete time). This filter may be non-causal, for example ``L = G^{-1}`` where ``G`` is the closed-loop transfer function.
- `location`: Either `:ref` or `:input`. If `:ref`, the ILC input is added to the reference signal, otherwise it is added to the input signal directly.
"""
@kwdef struct HeuristicILC{QT<:LTISystem{<:Discrete}, LT<:LTISystem{<:Discrete}} <: ILCAlgorithm
    Q::QT
    L::LT
    location::Symbol = :ref
end

function simulate(prob, alg::HeuristicILC, a)
    if alg.location === :ref
        lsim([prob.Gr prob.Gu], [prob.r+a; zero(a)])
    elseif alg.location === :input
        lsim([prob.Gr prob.Gu], [prob.r; a])
    else
        error("Unknown location: $(alg.location)")
    end
end

"""
    compute_input(prob, alg::ILCAlgorithm, workspace, a, e)

Compute the next ILC input using the learning rule

# Arguments:
- `alg`: The ILC algorithm
- `workspace`: A workspace created by calling [`init`](@ref) on the algorithm: `workspace = init(prob, alg)`.
- `a`: Previous ILC input
- `e`: Error `r - y`
"""
function compute_input(prob, alg::HeuristicILC, _, a, e)
    (; Q, L) = alg
    t = range(0, step=Q.Ts, length=size(a, 2))
    Le = lsim_noncausal(L, e, t)
    lsim_zerophase(Q, a + Le, t) # Update ILC adjustment
end


struct OptimizationILC <: ILCAlgorithm
    ρ::Float64
    λ::Float64
end
# NOTE: This algorithm can easily be extended to allow non-dentitity cost matrices, even to include frequency weighting in the cost.

"""
    OptimizationILC(; ρ = 1e-3, λ = 1e-3)

Optimization-based linear ILC algorithm from Norrlöf's thesis. This algorithm applies the ILC feedforward signal directly to the plant input.

# Arguments:
- `ρ`: Penalty on feedforward control action
- `λ`: Step size penalty
"""
function OptimizationILC(; ρ=1e-3, λ=1e-3)
    OptimizationILC(ρ, λ)
end

function init(prob, ::OptimizationILC)
    (; Tu = hankel_operator(prob.Gu, size(prob.r, 2)))
end

# TODO: adjust this algorithm to work with MIMO systems
function compute_input(prob, alg::OptimizationILC, workspace, a, e)
    (; ρ, λ) = alg
    Tu = workspace.Tu
    TTT = Symmetric(Tu'*Tu)
    Q = ((ρ+λ)*I + TTT)\(λ*I + TTT)
    L = (λ*I + TTT)\Tu'
    reshape(Q*(a' + L*hv(e)), reverse(size(a)))'
end

"""
    ilc(prob, alg; iters = 5, actual=prob)

Run the ILC algorithm for `iters` iterations. Returns a [`ILCSolution`](@ref) structure.
    
To manually perform ILC iterations, see the functions
- [`init`](@ref)
- [`compute_input`](@ref)

To simulate the effect of plant-model mismatch, one may provide a different instance of the ILCProblem using the `actual` keyword argument which is used to simulate the plant response. The ILC update will be performed using the plant model from `prob`, while simulated data will be acquired from the plant models in the `actual` problem.
"""
function ilc(prob, alg; iters = 5, actual = prob, kwargs...)
    workspace = init(prob, alg)
    r = prob.r
    a = zeros(nadjustment(prob), size(r, 2)) # ILC adjustment signal starts at 0
    Y = typeof(r)[]
    E = typeof(r)[]
    A = typeof(r)[]
    for iter = 1:iters
        res = simulate(actual, alg, a; kwargs...)
        y = res.y
        e = r .- y
        a = compute_input(prob, alg, workspace, a, e)
        push!(Y, y)
        push!(E, e)
        push!(A, a)
    end
    ILCSolution(Y,E,A,prob,alg)
end

@recipe function plot(sol::ILCSolution)
    layout := @layout([[a;b;c] d{0.3w}])
    rmses = sqrt.(sum.(abs2, sol.E) ./ length.(sol.E))

    @series begin
        label --> permutedims(["Iter $iter" for iter in 1:length(sol.Y)])
        sp --> 1
        reduce(vcat, sol.Y)'
    end
    
    @series begin
        sp --> 1
        title --> "Output \$y(t)\$"
        label --> "Reference \$r(t)\$"
        color --> :black
        linestyle --> :dash
        sol.prob.r'
    end

    @series begin
        title --> "Tracking error \$e(t)\$"
        sp --> 2
        legend --> false
        reduce(vcat, sol.E)'
    end

    @series begin
        title --> "Feedforward \$a(t)\$"
        sp --> 3
        legend --> false
        reduce(vcat, sol.A)'
    end

    @series begin
        title --> "Tracking RMS"
        legend --> false
        sp --> 4
        framestyle --> :zerolines
        rmses
    end
end

"""
    ilc_theorem(alg::HeuristicILC, Gc, Gcact = nothing)

Plot the stability boundary for the ILC algorithm.

# Arguments:
- `alg`: Containing the filters ``Q`` and ``L``
- `Gc`: The closed-loop system from ILC signal to output. If `alg.location = :ref`, this is typically given by `feedback(P*C)` while if `alg.location = :input`, this is typically given by `feedback(P, C)`.
- `Gcact`: If provided, this is the "actual" closed-loop system which may be constructed using a different plant model than `Gc`. This is useful when trying to determine if the filter choises will lead to a robust ILC algorithm. `Gc` may be constructed using, e.g., uncertain parameters, see https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/uncertainty/ for more details.
"""
function ilc_theorem(alg::HeuristicILC, Gc, Gcact=nothing; w=ControlSystemsBase._default_freq_vector(LTISystem[Gc, alg.L, alg.Q], Val(:bode)))
    (; L, Q) = alg
    fig = bodeplot(LTISystem[inv(Q), (1 - L*Gc)], w, plotphase=false, lab=["Stability boundary \$Q^{-1}\$" "\$1 - LG\$"], c=[:black 1], linestyle=[:dash :solid])
    fig2 = nyquistplot(Q*(1 - L*Gc), w, unit_circle=true, lab="\$Q(1 - LG)\$", ylims=(-2, 2), xlims=(-2, 2))
    if Gcact !== nothing
        bodeplot!(fig, (1 - L*Gcact), w, plotphase=false, lab="\$1 - LG\$ actual", c=2, q=1)
        nyquistplot!(fig2, Q*(1 - L*Gcact), w, lab="\$Q(1 - LG)\$ actual", ylims=(-1.5, 1.5), xlims=(-1.5, 1.5), q=1)
    end    
    RecipesBase.plot(fig, fig2)
end




"""
    ConstrainedILC(; Q, R, A, Y, Gr_constraints, Gu_constraints, opt, verbose=false, α)

Constrained ILC algorithm from the paper "On Robustness in Optimization-Based Constrained Iterative Learning Control", Liao-McPherson and friends.

The use of this ILC algorithms requires the user to manually install and load the packages `using JuMP, BlockArrays` as well as a compatible solver (such as `OSQP`).

Supports MIMO systems.

# Fields:
- `Q`: Error penalty matrix, e.g., `Q = I(ny)`
- `R`: Feedforward penalty matrix, e.g., `R = I(nu)`
- `A`: A function of `(model, a)` that adds constraints to the optimization problem. `a` is a size `(nu, N)` matrix of optimization variables that determines the optimized ILC input. See example below. 
- `Y`: A function of `(model, yh)` that adds constraints to the optimization problem. `yh` is a size `(ny, N)` matrix of predicted plant outputs. See example below
- `opt`: A JuMP-compatible optimizer, e.g., `OSQP.Optimizer`
- `α`: Step size, should be smaller than 2. Smaller step sizes lead to more robust progress but slower convergence. Use a small step size if the model is highly uncertain.
- `verbose`: If `true`, print solver output
- `Gr_constraints`: If provided, this is the closed-loop transfer function from reference to constrained outputs. If not provided, the constrained outputs are assumed to be equal to the plant outputs.
- `Gu_constraints`: If provided, this is the closed-loop transfer function from plant input to constrained outputs. If not provided, the constrained outputs are assumed to be equal to the plant outputs.

# Example
```
using IterativeLearningControl2, OSQP, JuMP, BlockArrays, ControlSystemsBase

# Define Gr and Gu

Q = 1000I(Gr.ny)
R = 0.001I(Gu.nu)

A = function (model, a) # Constrain the ILC input to the range [-25, 25]
    l,u = (-25ones(Gu.nu), 25ones(Gu.nu))
    JuMP.@constraint(model, [i=1:size(a, 2)], l .<= a[:, i] .<= u)
end

Y = function (model, yh) # Constrain the predicted output to the range [-1.1, 1.1]
    l,u = -1.1ones(Gr.ny), 1.1ones(Gr.ny)
    JuMP.@constraint(model, [i=1:size(yh, 2)], l .<= yh[:, i] .<= u)
end

alg = ConstrainedILC(; Q, R, A, Y, opt=OSQP.Optimizer, verbose=true, α=1)
```

To constrain the total plant input, i.e., the sum of the ILC feedforward and the output of the feedback controller, add outputs corresponding to this signal to the models `Gr, Gu`, for example
```
Gr_constraints = [Gr; feedback(C, P)]
Gu_constraints = [Gu; feedback(1, C*P)]
```
and constrain this output in the function `Y` above.
"""
@kwdef struct ConstrainedILC <: ILCAlgorithm
    Q
    R
    A = nothing
    Y = nothing
    Gr_constraints = nothing
    Gu_constraints = nothing
    α = nothing
    verbose = false
    opt
end


# function ΠWX(W, X, x)
#     l, u = X
#     # argmin_{v ∈ X} ||v-x||_W^2
#     opt = OSQP.Optimizer
#     model = JuMP.Model(opt)
#     JuMP.@variable(model, l[i] <= v[i=1:size(x,1), j=1:size(x,2)]<= u[i])
#     e = v .- x
#     JuMP.@objective(model, Min, dot(e, W, e))
#     JuMP.optimize!(model)
#     JuMP.value.(v)
# end

"""
A little helper function that takes a matrix with dimensions `(nsignals, n_timepoints)` and returns a reshaped vector version that is suitable for multiplying the Hankel operator obtained by calling [`hankel_operator`](@ref) or [`mv_hankel_operator`](@ref).
"""
hv(x) = vec(x')

"""
    mv_hankel_operator(sys::LTISystem{<:Discrete}, N::Int)

Return a matrix operator ``H`` such that `y == reshape(H*vec(u'), :, sys.ny)'` where `y = lsim(sys, u)`. ``H`` is a block Hankel matrix containing the Markov parameters of the system (scaled impulse response).

Use of this function requires the user to manually install and load the packages `using JuMP, BlockArrays`.
"""
function mv_hankel_operator end



## Nonlinear

"""
    ModelFreeILC

A model-free ILC scheme that works for linear, nonlinear systems and time-varying systems. The algorithm is described in the paper "Model-free Gradient Iterative Learning Control for Non-linear Systems" by Huo and friends.

!!! note
    This algorithm requires three rollouts per ILC iteration and is not yet compatible with [`compute_input`](@ref). When the ILCSolution is plotted, the number of actual rollouts performed is thus three times then number of ILC iterations shown in the plot (unless `store_all=true` in which case the true number of experiments is shown).


For non-square MIMO systems, this algorithm can only operate on the reference signal, i.e., with ``G_u = G_r`` in the ILCProblem. This is because the algorithm adds a scaled version of the tracking error to the ILC adjustment signal, and the two must thus be of compatible dimensions.

When this algorithm is used, an additional keyword argument to the function [`ilc`](@ref) is available, `store_all`. If set to `store_all = true`, the returned solution object will contain all intermediate plant responses, errors and ILC inputs. This is useful for debugging and plotting.

See also [`GradientILC`](@ref) for the model-based version of this algorithm. The model-based version only requires a single rollout per iteration.

# Fields:
- `α`: Perturbation size. An intermediate experiment with ``α e`` as input will be performed to compute the gradient. A value around 1 is a good start.
- `β`: Step size.
"""
@kwdef struct ModelFreeILC
    α::Float64
    β::Float64
end
# TODO: consider adding zero-phase filtering to suppress high-frequency noise in this algorithm
# TODO: consider extending the cost model with a penalty on control effort. It should be fairly straightforward, i.e., something like multiplying `a` by a number < 1, e.g., 0.99

function ilc(prob, alg::ModelFreeILC; iters = 5, actual = prob, store_all = false, kwargs...)
    workspace = init(prob, alg)
    (; α, β) = alg
    r = prob.r
    N = size(r, 2)
    a = zeros(nadjustment(prob), N) # ILC adjustment signal starts at 0
    Y = typeof(r)[]
    E = typeof(r)[]
    A = typeof(r)[]
    for iter = 1:iters

        # (1)
        y = simulate(actual, alg, a; kwargs...).y
        e = r .- y

        # (2)
        â = repeat(mean(a, dims=2), 1, N)
        ŷ = simulate(actual, alg, â; kwargs...).y
        if store_all
            push!(A, â)
            push!(Y, ŷ)
            push!(E, r .- ŷ)
        end

        # (3)
        ẽ = α*reverse(e, dims=2) + â
        ŷe = simulate(actual, alg, ẽ; kwargs...).y
        if store_all
            push!(A, ẽ)
            push!(Y, ŷe)
            push!(E, r .- ŷe)
        end

        # (4)
        a =  a .+ β/α .* reverse(ŷe - ŷ, dims=2)

        # a = compute_input(prob, alg, workspace, a, e)
        push!(Y, y)
        push!(E, e)
        push!(A, a)
    end
    ILCSolution(Y,E,A,prob,alg)
end

"""
    GradientILC(β)

A model-based gradient ILC scheme that works for linear or nonlinear systems. The ILC update rule is
```math
a_{k+1}(t) = a_k(t) + β H^T e_k(t)
```
where `H` is the Jacobian of the plant output w.r.t. ``a`` and `β` is the step size.

For non-square MIMO systems, this algorithm can only operate on the reference signal, i.e., with ``G_u = G_r`` in the ILCProblem. This is because the algorithm adds a scaled version of the tracking error to the ILC adjustment signal, and the two must thus be of compatible dimensions.

A model-free version of this algorithm is implemented in [`ModelFreeILC`](@ref).

# Fields:
- `β::Float64`: Step size (learning rate)
"""
struct GradientILC
    β::Float64
end
# TODO: consider adding zero-phase `Q` filter for more robust learning

function init(prob, alg::GradientILC)
    (; model = prob.Gu)
end

function compute_input(prob, alg::GradientILC, workspace, a, e)
    β = alg.β
    N = size(a, 2)
    model = workspace.model
    # TODO: consider implementing this as J'v without forming J explicitly
    a .+ reshape(β .* hankel_operator(model, N)' * hv(e), N, size(a, 1))'
end

include("LTVSystem.jl")

end


#=
# TODO: 
- Implement LTV ILC. This can probably be done for all existing algorithms. Filtering is replaced by a time-varying filter. Hankel operators are replaced by LTV hankel operators (page 106 (11) https://slunik.slu.se/kursfiler/TE0010/10095.1213/REG2_ILCReview.pdf)
- Introduce methods for `simulate`, `lsim_noncausal`, `hankel_operator` that works for LTV systems


# Real-world iterations
Implement a custom RealILCProblem that is passed as `actual`. The `simulate` function for this problem type should carry out the required experiment and return the result.
=#
