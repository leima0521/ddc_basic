## use NFP to solve a dynamic discrete choice model
## reference: https://github.com/jabbring/dynamic-discrete-choice
## Lei Ma, March 2021
using LinearAlgebra
using Random
using Distributions
using NLopt
using StatsPlots
cd();
cd("Dropbox/Github/ddc_basic")
include("ddc_func.jl")

## True parameters
T = 100;
N = 1000;
k = 5;
X = collect(1:5);
Π = 1 ./ (1 .+ abs.(ones(k, 1)*(1:k)' - collect(1:k)*ones(1, k)));
Π = Π./(sum(Π, dims = 1)'*ones(1,k));
β = [-0.1*k; 0.2];
δ = [0; 1];
ρ = 0.95;
Random.seed!(03022021);

(u0, u1) = flowpayoff(β, δ, X);
V0, V1 = zeros(k, 2), zeros(k, 2);
(V0, V1) = fixedpoint(V0, V1, u0, u1, Π, ρ);
Udiff = V1 .- V0;

## (1) Original data ϵ follows type I EV
(choices, Xi) = simulatedata(Udiff, Π, T, N);
opt = Opt(:LN_COBYLA, 3)
opt.xtol_rel = 1e-4
opt.max_objective = MLEObjFunc
@time (maxf,maxx,ret) = optimize(opt, [1; 1; 1])
hcat(vcat(β, δ[2]), maxx)

## (2) Misspecified data ϵ follows a normal dist
(choices, Xi) = simulatedata(Udiff, Π, T, N, "normal");
opt = Opt(:LN_COBYLA, 3)
opt.xtol_rel = 1e-4
opt.max_objective = MLEObjFunc
@time (maxf,maxx,ret) = optimize(opt, [1; 1; 1])
hcat(vcat(β, δ[2]), maxx)

## (3) Misspecified data ϵ follows a bimodal dist
#  check MCMC algorithm dist
q(x) = rand(Normal(x, 5), 1)[1];
p(x) = 0.5*exp(-0.7*(x-2)^2) + 0.5*exp(-0.7*(x+2)^2)[1];
ϵdist = mcmc(10000);
x = collect(range(-5, stop = 5, step = 0.001));
y = p.(x);
y_norm = y./sum(y)./0.001;
histogram(ϵdist, normalize =:pdf, label = "MCMC distribution", legend = (0.8, 0.9))
plot!(x, y_norm, label = "Target distribution", linewidth = 2.5)
savefig("mcmc_bimodal.png")

(choices, Xi) = simulatedata(Udiff, Π, T, N, "bimodal");
opt = Opt(:LN_COBYLA, 3)
opt.xtol_rel = 1e-4
opt.max_objective = MLEObjFunc
@time (maxf,maxx,ret) = optimize(opt, [1; 1; 1])
hcat(vcat(β, δ[2]), maxx)
