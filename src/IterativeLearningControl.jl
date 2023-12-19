module IterativeLearningControl
using ControlSystemsBase, RecipesBase, LinearAlgebra

export ilc,
    OptimizationILC, HeuristicILC,
    ILCProblem,
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

function hankel(x,L::Int)
    H = zeros(L, L)
    for i = 1:L
        H[i:end, i] = x[1:end-i]
    end
    H
end

function hankel(sys::LTISystem{<:Discrete}, N::Int)
    ControlSystemsBase.issiso(sys) || error("System must be SISO")
    Tf = N*sys.Ts
    imp = impulse(sys, Tf).y[:] .* sys.Ts # TODO: test with non-unit Ts
    hankel(imp, N)
end

struct ILCSolution
    Y
    E
    A
    prob
    alg
end
struct ILCProblem
    r
    sim
end

simulate(prob, alg, a) = prob.sim(prob.r, a)

abstract type ILCAlgorithm end

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
$$| 1 - LG | < |Q^{-1}|$$
which we can verify by looking at the Bode curves of the two sides of the inequality
```@example ilc
bodeplot([inv(Q), (1 - L*Gc)], plotphase=false, lab=["Stability boundary \$Q^{-1}\$" "\$1 - LG\$"])
bodeplot!((1 - L*Gcact), plotphase=false, lab="\$1 - LG\$ actual")
```
This plot can be constructed using the [`ilc_theorem`](@ref) function.

# Fields:
- `Q`: Robustness filter
- `L`: Learning filter
- `t`: Time vector
"""
@kwdef struct HeuristicILC <: ILCAlgorithm
    Q
    L
    t
    location::Symbol = :ref
end

function simulate(prob, alg::HeuristicILC, a)
    if alg.location === :ref
        prob.sim(prob.r+a, zero(a))
    elseif alg.location === :input
        prob.sim(prob.r, a)
    else
        error("Unknown location: $(alg.location)")
    end
end

function compute_input(alg::HeuristicILC, a, e)
    (; Q, L, t) = alg
    Le = lsim_noncausal(L, e, t)
    lsim_zerophase(Q, a + Le, t) # Update ILC adjustment
end


struct OptimizationILC{T} <: ILCAlgorithm
    Tu::T
    ρ::Float64
    λ::Float64
end

"""
    OptimizationILC(Gu::LTISystem; N::Int, ρ = 1e-3, λ = 1e-3)

Optimization-based linear ILC algorithm from Norrlöf's thesis.

# Arguments:
- `Gu`: System model from ILC feedforward input to output
- `N`: Trajectory length
- `ρ`: Penalty on feedforward control action
- `λ`: Step size penalty
"""
function OptimizationILC(Gu::LTISystem; N::Int, ρ=1e-3, λ=1e-3)
    Tu = hankel(Gu, N)
    OptimizationILC(Tu, ρ, λ)
end

function compute_input(alg::OptimizationILC, a, e)
    (; Tu, ρ, λ) = alg
    TTT = Tu'*Tu
    Q = ((ρ+λ)*I + TTT)\(λ*I + TTT)
    L = (λ*I + TTT)\Tu'
    a = (Q*(a' + L*e'))'
end

function ilc(prob, alg; iters = 5)
    r = prob.r
    a = zero(r) # ILC adjustment signal starts at 0
    Y = typeof(r)[]
    E = typeof(r)[]
    A = typeof(r)[]
    for iter = 1:iters
        res = simulate(prob, alg, a)
        y = res.y
        e = r .- y
        a = compute_input(alg, a, e)
        push!(Y, y)
        push!(E, e)
        push!(A, a)
    end
    ILCSolution(Y,E,A,prob,alg)
end

@recipe function plot(sol::ILCSolution)

    layout := @layout([[a;b;c] d{0.3w}])
    @series begin
        title --> ["Output \$y(t)\$" "Feedforward \$a\$"]
        label --> permutedims(["Iter $iter" for iter in 1:length(sol.Y)])
        sp --> 1
        reduce(vcat, sol.Y)'
    end

    rmses = sqrt.(sum.(abs2, sol.E) ./ length.(sol.E))


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
