# the deterministic part of flow utility
function flowpayoff(β, δ, X)
    (β0, β1) = β
    (δ0, δ1) = δ
    k = size(X, 1)
    u0 = [zeros(k, 1)  -δ0*ones(k, 1)]
    u1 = [β0*ones(k, 1) .+ β1*X .- δ1  β0*ones(k, 1).+β1*X]
    return (u0, u1)
end

function bellman(V0, V1, u0, u1, Π, ρ)
    # expected value at t+1 (not exactly) for a_t = 0 and a_t = 1
    EV0 = log.(exp.(V0[:, 1]) + exp.(V1[:, 1]))
    EV1 = log.(exp.(V0[:, 2]) + exp.(V1[:, 2]))
    # bellman equation
    V0 = u0 + ρ.*Π*EV0*[1 1]
    V1 = u1 + ρ.*Π*EV1*[1 1]
    return (V0, V1)
end

function fixedpoint(V0, V1, u0, u1, Π, ρ, tol = 1e-10, maxiter = 1e5)
    V0_old, V1_old = V0, V1
    normdiff = Inf
    iter = 0
    while normdiff > tol && iter <= maxiter
        (V0_new, V1_new) = bellman(V0_old, V1_old, u0, u1, Π, ρ)
        normdiff0 = maximum(norm.(V0_new .- V0_old))
        normdiff1 = maximum(norm.(V1_new .- V1_old))
        normdiff = maximum([normdiff0, normdiff1])
        V0_old, V1_old = V0_new, V1_new
        iter += 1
    end
    return (V0_old, V1_old)
end

function randdiscrete(p)
    k = size(p, 1)
    N = size(p, 2)
    unifdraws = ones(k-1, 1)*rand(Uniform(), N)'
    cumpinf = cumsum(p, dims = 1)
    Xi = sum([ones(1, N); cumpinf[1:k-1, :] .<= unifdraws], dims = 1)
end

# simulate data (X_t, a_t)
function simulatedata(Udiff, Π, T, N)
    k = size(Π, 1)
    one_minus_pi = Matrix(1.0I, k, k) - Π'
    pinf = [one_minus_pi[1:k-1, :]; ones(1, k)]\[zeros(k-1, 1); 1.0]
    pinf = pinf * ones(1, N)

    Xi = randdiscrete(pinf)
    ϵdiff = rand(Gumbel(), N) - rand(Gumbel(), N)
    choices = Udiff[Array{Int64,2}(Xi), 1] .> ϵdiff'

    for t = 2:T
        Xi = [Xi; randdiscrete(Π[Array{Int64,1}(Xi[end,:]), :]')]
        ϵdiff = rand(Gumbel(), N) - rand(Gumbel(), N)
        choices = [choices; (Udiff[Array{Int64,1}(Xi[end,:]).+ k*choices[end,:]])' .> ϵdiff']
    end
    return (choices, Xi)
end

# Estimate the state transition matrix from data
function state_transition(Xi, k)
    T = size(Xi, 1)
    Π_est = zeros(k, k)
    for i = 1:k
        for j = 1:k
          Π_est[i, j] = sum((Xi[2:T, :] .== j) .& (Xi[1:T-1, :] .== i))/sum(Xi[1:T-1, :] .== i)
        end
    end
    return Π_est
end

# MLE
function MLEObjFunc(theta::Vector, grad::Vector)
    β, δ = theta[1:2], theta[3]
    δ = [0; δ]
    (u0, u1) = flowpayoff(β, δ, X)
    Π = state_transition(Xi, k)
    V0, V1 = zeros(k, 2), zeros(k, 2)
    (V0, V1) = fixedpoint(V0, V1, u0, u1, Π, ρ)
    Udiff = V1 .- V0
    pexit = 1 ./ (1 .+ exp.(Udiff))

    lagged_choice = [zeros(1, size(choices, 2)); choices[1:end-1, :]]
    p = choices .+ (1 .- 2*choices) .* pexit[Array{Int64}(Xi + k*lagged_choice)]
    ll = sum(log.(p))
    return ll
end
