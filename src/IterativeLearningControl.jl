module IterativeLearningControl
using ControlSystemsBase, RecipesBase, LinearAlgebra

export ilc,
    OptimizationILC, HeuristicILC,
    ILCProblem


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
struct HeuristicILC <: ILCAlgorithm
    Q
    L
    t
end

simulate(prob, alg::HeuristicILC, a) = prob.sim(prob.r+a, zero(a))

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
    # fig1 = plot(t, vec(r), sp=1, layout=(3,1), l=(:black, 3), lab="Ref")
    # fig2 = plot(title="Sum of squared error", xlabel="Iteration", legend=false, titlefontsize=10, framestyle=:zerolines, ylims=(0, 7.1))
    Y = []
    E = []
    A = []
    for iter = 1:iters
        res = simulate(prob, alg, a)         # System response
        y = res.y
        e = r .- y            # Error
        a = compute_input(alg, a, e)
        err = sum(abs2, e)
        # plot!(fig1, res, plotu=true, sp=[1 2 2], title=["Output \$y(t)\$" "Feedforward \$a\$"], lab="Iter $iter", c=iter)
        # plot!(fig1, res.t, e[:], sp=3, title="Tracking error \$e(t)\$", lab="err: $(round(err, digits=2))", c=iter)
        # scatter!(fig2, [iter], [err])
        push!(Y, y)
        push!(E, e)
        push!(A, a)
    end
    # plot(fig1, fig2, layout=@layout([a{0.7w} b{0.3w}]))
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
        rmses
    end



end


end
