import ControlSystemsBase: LTISystem, Discrete, Continuous, TimeEvolution, output_names, input_names, state_names, numeric_type, SimResult, linearize, AbstractSystem, feedback
import Base: +, -, *, /, ^, hcat, vcat
using ForwardDiff

export LTVSystem, NonlinearSystem, tcat, NonlinearILCProblem

struct LTVSystem{T, N} <: AbstractSystem
    A::Array{T, 3}
    B::Array{T, 3}
    C::Array{T, 3}
    D::Array{T, 3}
    Ts::Float64
end

LTVSystem(A::Array{T,3},B::Array{T,3},C::Array{T,3},D::Array{T,3},Ts) where T =
    LTVSystem{T, size(A, 3)}(A, B, C, D, Ts)

function Base.getproperty(sys::LTVSystem{<:Any, N}, s::Symbol) where N
    s ∈ fieldnames(typeof(sys)) && return getfield(sys, s)
    if s === :N
        return N
    elseif s === :nx
        return size(sys.A, 1)
    elseif s === :nu
        return size(sys.B, 2)
    elseif s === :ny
        return size(sys.C, 1)
    else
        throw(ArgumentError("$(typeof(sys)) has no property named $s"))
    end
end


function LTVSystem(syss::Vector{<:LTISystem{<:Discrete}})
    N = length(syss)
    (; nx, nu, ny) = syss[1]
    T = numeric_type(syss[1])
    A = zeros(T, nx, nx, N)
    B = zeros(T, nx, nu, N)
    C = zeros(T, ny, nx, N)
    D = zeros(T, ny, nu, N)
    for i = 1:N
        A[:, :, i] .= syss[i].A
        B[:, :, i] .= syss[i].B
        C[:, :, i] .= syss[i].C
        D[:, :, i] .= syss[i].D
    end
    LTVSystem{T, N}(A, B, C, D, syss[1].Ts)
end

Base.length(m::LTVSystem{<:Any, N}) where N = N
nadjustment(m::LTVSystem) = m.nu
Base.getindex(m::LTVSystem, i::Integer) = ss(m.A[:,:,i], m.B[:,:,i], m.C[:,:,i], m.D[:,:,i], m.Ts)
Base.getindex(m::LTVSystem, i::Integer, j::Integer) = LTVSystem(m.A[i:i, j:j, :], m.B[i:i, j:j, :], m.C[i:i, j:j, :], m.D[i:i, j:j, :], m.Ts)
Base.size(m::LTVSystem, i::Integer) = size(m.D, i)

timevec(sys::LTVSystem) = range(0, step=sys.Ts, length=sys.N)

function hankel_operator(sys::LTVSystem, N::Int=length(sys))
    N == length(sys) || error("The number of systems must be equal to N")
    u = zeros(sys.nu, N)
    ControlSystemsBase.ForwardDiff.jacobian(hv(u)) do u
        hv(lsim(sys, reshape(u, :, sys.nu)').y)
    end
end

@views function ControlSystemsBase.lsim(sys::LTVSystem, u::AbstractArray{T}; x0 = zeros(sys.nx)) where T
    N = length(sys)
    size(u) == (sys.nu, N) || error("The input-signal array u must have size (nu, N) where N is the number of systems and nu is the number of inputs to those systems")
    (; nx, ny) = sys
    y = zeros(T, ny, N)
    x = zeros(T, nx, N)
    x[:, 1] .= x0
    for i = 1:N
        # y[:, i] = sys.C[:,:,i] * x[:, i] + sys.D[:,:,i] * u[:, i]
        mul!(y[:, i], sys.C[:,:,i], x[:, i])
        mul!(y[:, i], sys.D[:,:,i], u[:, i], 1, 1)
        if i < N
            # x[:, i+1] = sys.A[:,:,i] * x[:, i] + sys.B[:,:,i] * u[:, i]
            mul!(x[:, i+1], sys.A[:,:,i], x[:, i])
            mul!(x[:, i+1], sys.B[:,:,i], u[:, i], 1, 1)
        end
    end
    t = range(0, step=sys.Ts, length=N)
    SimResult(y, t, x, u, sys)
end

ControlSystemsBase.iscontinuous(::LTVSystem) = false
ControlSystemsBase.isdiscrete(::LTVSystem) = true
ControlSystemsBase.output_names(syss::LTVSystem, args...) = output_names(first(syss), args...)
ControlSystemsBase.input_names(syss::LTVSystem, args...) = input_names(first(syss), args...)
ControlSystemsBase.state_names(::LTVSystem, args...) = state_names(first(syss), args...)
ControlSystemsBase.numeric_type(::Type{LTVSystem{T}}) where T = T

function Base.promote_type(::Type{LTVSystem{T,N}}, LTI::Type{<:AbstractStateSpace{<:Discrete}}) where {T,N}
    LTVSystem{promote_type(T, numeric_type(LTI)), N}
end

function Base.convert(::Type{LTVSystem{T,N}}, sys::LTISystem{<:Discrete}) where {T,N}
    LTVSystem(fill(sys, N))
end

function Base.isapprox(s1::LTVSystem{Float64}, s2::LTVSystem{Float64}; kwargs...)
    s1.Ts == s2.Ts && s1.N == s2.N &&
    isapprox(s1.A, s2.A; kwargs...) &&
    isapprox(s1.B, s2.B; kwargs...) &&
    isapprox(s1.C, s2.C; kwargs...) &&
    isapprox(s1.D, s2.D; kwargs...)
end

function systemwise(f, sys::LTVSystem, args...; kwargs...)
    N = length(sys)
    map(1:N) do i
        f(sys[i], args...; kwargs...)
    end |> LTVSystem
end

function systemwise(f, arg, sys::LTVSystem, args...; kwargs...)
    N = length(sys)
    map(1:N) do i
        f(arg, sys[i], args...; kwargs...)
    end |> LTVSystem
end

function systemwise(f, sys1::LTVSystem, sys2::LTVSystem, args...; kwargs...)
    N = length(sys1)
    N == length(sys2) || error("The number of systems must be equal")
    map(1:N) do i
        f(sys1[i], sys2[i], args...; kwargs...)
    end |> LTVSystem
end

for fun in [:feedback, :*, :+, :-, :/, :^]
    @eval $fun(sys::LTVSystem, args...; kwargs...) = systemwise($(fun), sys, args...; kwargs...)
    @eval $fun(arg, sys::LTVSystem, args...; kwargs...) = systemwise($(fun), arg, sys, args...; kwargs...)
    @eval $fun(sys1::LTVSystem, sys2::LTVSystem, args...; kwargs...) = systemwise($(fun), sys1, sys2, args...; kwargs...)
end

Base.hcat(sys1::LTVSystem, sys2::LTVSystem) = systemwise(Base.hcat, sys1, sys2)
Base.vcat(sys1::LTVSystem, sys2::LTVSystem) = systemwise(Base.vcat, sys1, sys2)

function tcat(s1::LTVSystem{T,N1}, s2::LTVSystem{T,N2}) where {T,N1,N2}
    s1.Ts == s2.Ts || error("The sampling times must be equal")
    LTVSystem{T, N1+N2}(cat(s1.A, s2.A, dims=3), cat(s1.B, s2.B, dims=3), cat(s1.C, s2.C, dims=3), cat(s1.D, s2.D, dims=3), s1.Ts)
end

@kwdef struct NonlinearSystem{F, G}
    f::F
    g::G
    nx::Int
    nu::Int
    ny::Int
    na::Int
    Ts::Float64
end

nadjustment(sys::NonlinearSystem) = sys.na

function linearize(m::NonlinearSystem, x0::AbstractVector, u0::AbstractVector, args...)
    A, B = linearize(m.f, x0, u0, args...)
    C, D = linearize(m.g, x0, u0, args...)
    ss(A, B, C, D, m.Ts)
end

function linearize!(f, A, B, va::AbstractVector, vb::AbstractVector, xi::AbstractVector, ui::AbstractVector, args...)
    ForwardDiff.jacobian!(A, x -> f(x, ui, args...), xi)
    ForwardDiff.jacobian!(B, u -> f(xi, u, args...), ui)
    A, B
end

function linearize(m::NonlinearSystem, x0::AbstractMatrix, a0::AbstractMatrix, r, p, args...)
    (; nx, nu, ny) = m
    N = size(x0, 2)
    size(a0, 2) == size(r, 2) == N || error("The length of the input-signal arrays u and r must be equal to the length of the state-signal array x, got $(size(a0, 2)) and $(size(r, 2)) respectively")
    size(a0, 1) == nu || error("The number of rows in the input-signal array u must be equal to the number of inputs to the system ($(nu))")
    size(x0, 1) == nx || error("The number of rows in the state-signal array x must be equal to the state dimension $(sys.nx) in the system")
    A = zeros(nx, nx, N)
    B = zeros(nx, nu, N)
    C = zeros(ny, nx, N)
    D = zeros(ny, nu, N)
    va = zeros(nx)
    vb = zeros(nx)
    vc = zeros(ny)
    vd = zeros(ny)
    @views for i = 1:N
        t = (i-1)*m.Ts
        linearize!(m.f, A[:, :, i], B[:, :, i], va, vb, x0[:, i], a0[:, i], r[:, i], p, t, args...)
        linearize!(m.g, C[:, :, i], D[:, :, i], vc, vd, x0[:, i], a0[:, i], r[:, i], p, t, args...)
    end
    LTVSystem(A, B, C, D, m.Ts)

end

@kwdef struct NonlinearILCProblem
    r
    model
    x0
end

nadjustment(prob::NonlinearILCProblem) = nadjustment(prob.model)

# TODO: it seems we have to handle the fact that we have reference inputs as well, i.e., rewrite linearize to such that it expects `f,g` to have the signatures f(x, a, r, p, t)?

function simulate(prob::NonlinearILCProblem, alg, a; p=nothing)
    model = prob.model
    r = prob.r
    (; nx, ny) = model
    N = size(a, 2)
    x = zeros(nx, N)
    y = zeros(ny, N)
    x[:, 1] .= prob.x0
    @views for i = 1:N
        y[:, i] = model.g(x[:, i], a[:, i], r[:, i], p, (i-1)*model.Ts)
        if i < N
            x[:, i+1] = model.f(x[:, i], a[:, i], r[:, i], p, (i-1)*model.Ts)
        end
    end
    t = range(0, step=model.Ts, length=N)
    SimResult(y, t, x, a, model)
end


function init(prob::NonlinearILCProblem, ::OptimizationILC)
    nothing
end

function compute_input(prob::NonlinearILCProblem, alg::OptimizationILC, workspace, a, e, p=nothing)
    (; ρ, λ) = alg
    r = prob.r
    traj = simulate(prob, alg, a; p)
    ltv = linearize(prob.model, traj.x, traj.u, r, p)
    Tu = hankel_operator(ltv)
    
    TTT = Symmetric(Tu'*Tu)
    Q = ((ρ+λ)*I + TTT)\(λ*I + TTT)
    L = (λ*I + TTT)\Tu'
    reshape(Q*(a' + L*hv(e)), reverse(size(a)))'
end

function init(prob::NonlinearILCProblem, ::GradientILC)
    (; model = prob.model)
end

function compute_input(prob::NonlinearILCProblem, alg::GradientILC, workspace, a, e, p=nothing)
    β = alg.β
    N = size(a, 2)
    r = prob.r
    model = workspace.model
    traj = simulate(prob, alg, a; p)
    ltv = linearize(model, traj.x, traj.u, r, p)
    Tu = hankel_operator(ltv)
    # TODO: consider implementing this as J'v without forming J explicitly
    a .+ reshape(β .* Tu' * hv(e), N, size(a, 1))'
end