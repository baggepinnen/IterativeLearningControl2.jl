module IterativeLearningControlJuMP
using IterativeLearningControl, ControlSystemsBase, JuMP, LinearAlgebra, BlockArrays

import IterativeLearningControl: mv_hankel_operator, hankel_operator, hv, compute_input, init, simulate, linearize



"""
    mv_hankel_operator(sys::LTISystem{<:Discrete}, N::Int)

Return a matrix operator ``H`` such that `y == reshape(H*vec(u'), :, sys.ny)'` where `y = lsim(sys, u)`. ``H`` is a block Hankel matrix containing the Markov parameters of the system (scaled impulse response).
"""
function mv_hankel_operator(sys::LTISystem{<:Discrete}, N::Int)
    Hs = [hankel_operator(sys[i, j], N) for i in 1:size(sys, 1), j in 1:size(sys, 2)]
    BlockArrays.mortar(Hs) |> Matrix # Call to Matrix due to https://github.com/JuliaArrays/BlockArrays.jl/issues/325
end

function compute_input(prob, alg::ConstrainedILC, workspace, a, e)
    (; A, Y, verbose, opt) = alg
    (; w, Mz, Mv, W, Q, R) = workspace

    α = something(alg.α, workspace.α)
    α < 2 || @warn "α = $α is too large, should be < 2" maxlog=1

    model = JuMP.Model(opt)
    JuMP.set_optimizer_attribute(model, JuMP.MOI.Silent(), !verbose)
    JuMP.@variable(model, v[i=1:size(a,1), j=1:size(a,2)])
    A !== nothing && A(model, v)
    eu = hv(v .- a)
    F̄ = Mz'Q*hv(-e) + R*hv(a)

    JuMP.@objective(model, Min, eu'W*eu + α*(hv(v)'*F̄))
    yh = reshape(Mv*hv(v) + w, size(a, 2), :)'
    Y !== nothing && Y(model, yh) # Adds Y constraints
    JuMP.optimize!(model)
    vv = JuMP.value.(v)
    all(isfinite, vv) || error("Solution is not finite, the problem may be infeasible")
    reshape(vv, size(a, 2), size(a, 1))'

end

function init(prob, alg::ConstrainedILC)
    (; Q, R) = alg

    # z denotes performance outputs / controlled outputs, while v denotes constrained outputs

    N = size(prob.r, 2)
    Mz = hankel_operator(prob.Gu, N)
    if alg.Gu_constraints === nothing
        Mv = Mz # We constrain the same outputs as the performance outputs
    else
        Mv = hankel_operator(alg.Gu_constraints, N) # We have special constrained outputs
    end
    QB = kron(I(N), Q)
    RB = kron(I(N), R)

    W = Mz'QB*Mz + RB # Penalty is associated with controlled outputs

    # Smart step-size selection (requires access to polytope models Gi)
    # Wc = sqrt(Symmetric(W))
    # H = 0.5 .* (M'QB*Gi + RB) # NOTE: should be H = M*QB*Gi' + RB
    # Θ = Symmetric(Wc\(H + H')/Wc)
    # evs = eigvals(Θ)
    # μ, L = extrema(evs)
    # α = μ / L^2

    α = 0.5 # stepsize

    Gr = something(alg.Gr_constraints, prob.Gr)

    (;
        Mz,
        Mv,
        w = hv(lsim(Gr, prob.r).y), # constrained reference response
        W,
        Q = QB,
        R = RB,
        α,
    )
end

function init(prob::NonlinearILCProblem, alg::ConstrainedILC)
    alg.Gu_constraints == alg.Gr_constraints == nothing || error("Nonlinear ILC with constrained outputs differing from the plant output is not yet supported")
    (; Q, R) = alg


    N = size(prob.r, 2)
    QB = kron(I(N), Q)
    RB = kron(I(N), R)
    α = 0.5 # stepsize
    (;
        Q = QB,
        R = RB,
        α,
    )
end



function compute_input(prob::NonlinearILCProblem, alg::ConstrainedILC, workspace, a, e, p=prob.p)
    (; A, Y, verbose, opt) = alg
    (; Q, R) = workspace
    r = prob.r

    α = something(alg.α, workspace.α)
    α < 2 || @warn "α = $α is too large, should be < 2" maxlog=1

    model = JuMP.Model(opt)
    JuMP.set_optimizer_attribute(model, JuMP.MOI.Silent(), !verbose)
    JuMP.@variable(model, v[i=1:size(a,1), j=1:size(a,2)])
    A !== nothing && A(model, v)
    eu = hv(v .- a)

    traj = simulate(prob, alg, a; p)
    w = hv(traj.y)
    ltv = linearize(prob.model, traj.x, traj.u, r, p)
    Mz = hankel_operator(ltv)
    W = Mz'Q*Mz + R
    Mv = Mz
    F̄ = Mz'Q*hv(-e) + R*hv(a)

    JuMP.@objective(model, Min, eu'W*eu + α*(hv(v)'*F̄))
    yh = reshape(Mv*hv(v) + w, size(a, 2), :)'
    Y !== nothing && Y(model, yh) # Adds Y constraints
    JuMP.optimize!(model)
    vv = JuMP.value.(v)
    all(isfinite, vv) || error("Solution is not finite, the problem may be infeasible")
    reshape(vv, size(a, 2), size(a, 1))'

end

end