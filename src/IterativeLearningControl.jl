module IterativeLearningControl
using ControlSystemsBase, RecipesBase, LinearAlgebra

export ilc,
    OptimizationILC, HeuristicILC,
    ILCProblem,
    init, compute_input,
    ilc_theorem


function lsim_zerophase(G, u, args...; kwargs...)
    res = lsim(G, u[:, end:-1:1], args...; kwargs...)
    lsim(G, res.y[:, end:-1:1], args...; kwargs...).y
end

function lsim_noncausal(L::LTISystem{<:Discrete}, u, args...; kwargs...)
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

"""
    hankel(sys::LTISystem{<:Discrete}, N::Int)

Return a matrix operator ``H`` such that ``Hu^T = y^T`` where `y = lsim(H, u)`. `H` is a Hankel matrix containing the Markov parameters of the system (scaled impulse response).
"""
function hankel_operator(sys::LTISystem{<:Discrete}, N::Int)
    ControlSystemsBase.issiso(sys) || error("System must be SISO")
    Tf = N*sys.Ts
    imp = impulse(sys, Tf).y[:] .* sys.Ts
    hankel_operator(imp, N)
end

"""
    ILCSolution

A structure representing the solution to an ILC problem. 

# Fields:
- `Y`: Plant responses. `Y[i]` is the response during the `i`th iteration
- `E`: Errors. `E[i]` is the error during the `i`th iteration
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
- `r`: Reference signal
- `Gr`: Closed-loop transfer function from reference to output
- `Gu`: Closed-loop transfer function from plant input to output
"""
struct ILCProblem
    r
    Gr
    Gu
end


"""
    ILCProblem(; r, Gr, Gu)
    ILCProblem(; r, P, C)

Construct an ILCProblem given either
- The closed-loop transfer functions from reference to output and from plant input to output, or
- The plant and controller transfer functions


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
    N = size(r, 2)
    t = range(0, step=Gr.Ts, length=N)
    ILCProblem(r, Gr, Gu)
end

simulate(prob, alg, a) = lsim([prob.Gr prob.Gu], [prob.r; a])

abstract type ILCAlgorithm end

"""
    init(prob, alg)

Initialize the ILC algorithm. This function is called internally by the funciton [`ilc`](@ref) but manual iterations require the user to initialize the workspace explicitly.
"""
init(prob, alg) = nothing

"""
    HeuristicILC

Apply the learning rule

```math
\\begin{aligned}
y_k(t) &= G(q) \\big(r(t) + a_k(t) \\big) \\\\
e_k(t) &= r(t) - y_k(t) \\\\
a_k(t) &= Q(q) \\big( a_{k-1}(t) + L(q) e_{k-1}(t) \\big)
\\end{aligned}
```

A theorem due to Norrlöf says that for this ILC iterations to converge, one needs to satisfy
```math
| 1 - LG | < |Q^{-1}|
```
which we can verify by looking at the Bode curves of the two sides of the inequality
```@example ilc
bodeplot([inv(Q), (1 - L*Gc)], plotphase=false, lab=["Stability boundary \$Q^{-1}\$" "\$1 - LG\$"])
bodeplot!((1 - L*Gcact), plotphase=false, lab="\$1 - LG\$ actual")
```
This plot can be constructed using the [`ilc_theorem`](@ref) function.

# Fields:
- `Q`: Robustness filter. The filter will be applied both forwards and backwards in time (like `filtfilt`), and the effective filter transfer funciton is thus ``Q(z)Q(z̄)``.
- `L`: Learning filter. This filter may be non-causal, for example `L = G^{-1}` where ``G`` is the closed-loop transfer function.
- `location`: Either `:ref` or `:input`. If `:ref`, the ILC input is added to the reference signal, otherwise it is added to the input signal directly.
"""
@kwdef struct HeuristicILC <: ILCAlgorithm
    Q
    L
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
    compute_input(alg::ILCAlgorithm, workspace, a, e)

Compute the next ILC input using the learning rule

# Arguments:
- `alg`: The ILC algorithm
- `workspace`: A workspace created by calling [`init`](@ref) on the algorithm: `workspace = init(prob, alg)`.
- `a`: Previous ILC input
- `e`: Error `r - y`
"""
function compute_input(alg::HeuristicILC, _, a, e)
    (; Q, L) = alg
    t = range(0, step=Q.Ts, length=size(a, 2))
    Le = lsim_noncausal(L, e, t)
    lsim_zerophase(Q, a + Le, t) # Update ILC adjustment
end


struct OptimizationILC <: ILCAlgorithm
    ρ::Float64
    λ::Float64
end

"""
    OptimizationILC(Gu::LTISystem; ρ = 1e-3, λ = 1e-3)

Optimization-based linear ILC algorithm from Norrlöf's thesis. This algorithm applies the ILC feedforward signal directly to the plant input.

# Arguments:
- `ρ`: Penalty on feedforward control action
- `λ`: Step size penalty
- `Gu`: System model from ILC feedforward input to output
"""
function OptimizationILC(; ρ=1e-3, λ=1e-3)
    OptimizationILC(ρ, λ)
end

function init(prob, alg::OptimizationILC)
    (; Tu = hankel_operator(prob.Gu, size(prob.r, 2)))
end

function compute_input(alg::OptimizationILC, workspace, a, e)
    (; ρ, λ) = alg
    Tu = workspace.Tu
    TTT = Tu'*Tu
    Q = ((ρ+λ)*I + TTT)\(λ*I + TTT)
    L = (λ*I + TTT)\Tu'
    (Q*(a' + L*e'))'
end

"""
    ilc(prob, alg; iters = 5, actual=prob)

Run the ILC algorithm for `iters` iterations. Returns a [`ILCSolution`](@ref) structure.

To simulate the effect of plat-model mismatch, one may provide a different instance of the ILCProblem using the `actual` keyword argument which is used to simulate the plant response. The ILC update will be performed using the plant model from `prob`, while simulated data will be aquired from `actual`.
"""
function ilc(prob, alg; iters = 5, actual = prob)
    workspace = init(prob, alg)
    r = prob.r
    a = zero(r) # ILC adjustment signal starts at 0
    Y = typeof(r)[]
    E = typeof(r)[]
    A = typeof(r)[]
    for iter = 1:iters
        res = simulate(actual, alg, a)
        y = res.y
        e = r .- y
        a = compute_input(alg, workspace, a, e)
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
        title --> ["Output \$y(t)\$" "Feedforward \$a\$"]
        label --> permutedims(["Iter $iter" for iter in 1:length(sol.Y)])
        sp --> 1
        reduce(vcat, sol.Y)'
    end

    @series begin
        title --> "Tracking error \$e(t)\$"
        sp --> 2
        legend --> false
        reduce(vcat, sol.E)'
    end

    @series begin
        title --> "Feedforward\$"
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
function ilc_theorem(alg::HeuristicILC, Gc, Gcact=nothing)
    (; L, Q) = alg
    fig = bodeplot([inv(Q), (1 - L*Gc)], plotphase=false, lab=["Stability boundary \$Q^{-1}\$" "\$1 - LG\$"], c=[:black 1], linestyle=[:dash :solid])
    fig2 = nyquistplot(Q*(1 - L*Gc), unit_circle=true, lab="\$Q(1 - LG)\$")
    if Gcact !== nothing
        bodeplot!(fig, (1 - L*Gcact), plotphase=false, lab="\$1 - LG\$ actual", c=2)
        nyquistplot!(fig2, Q*(1 - L*Gcact), lab="\$Q(1 - LG)\$ actual")
    end    
    RecipesBase.plot(fig, fig2)
end


end
