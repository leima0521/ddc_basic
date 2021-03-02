## use NFP to solve a dynamic discrete choice model
## reference: https://github.com/jabbring/dynamic-discrete-choice
## Lei Ma, Feb 2021
using LinearAlgebra
using Random
using Distributions
using NLopt
using StatsPlots
cd();
cd("Dropbox/Boston University/Spring 2021/Econometrics/Project")
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

#### Simulate data using true parameters
(u0, u1) = flowpayoff(β, δ, X);
V0, V1 = zeros(k, 2), zeros(k, 2);
(V0, V1) = fixedpoint(V0, V1, u0, u1, Π, ρ);
Udiff = V1 .- V0;

## (1) Original data ϵ follows type I EV
(choices, Xi) = simulatedata(Udiff, Π, T, N);
##     Estimate parameters
opt = Opt(:LN_COBYLA, 3)
opt.xtol_rel = 1e-4
opt.max_objective = MLEObjFunc
@time (maxf,maxx,ret) = optimize(opt, [1; 1; 1])
hcat(vcat(β, δ[2]), maxx)

## (2) Misspecified data ϵ follows a bimodal dist
#  check MCMC algorithm dist
q(x) = rand(Normal(x, 10), 1)[1];
p(x) = 0.3*exp(-0.2*x^2) + 0.7*exp(-0.2*(x-10)^2)[1];
ϵdist = mcmc(10000);
x = collect(range(-5, stop = 15, step = 0.001));
y = p.(x);
y_norm = y./sum(y)./0.001;
histogram(ϵdist, normalize =:pdf, label = "MCMC distribution", legend = (0.15, 0.9))
plot!(x, y_norm, label = "Target distribution", linewidth = 2.5)
savefig("mcmc_bimodal.png")

##     Estimate parameters
(choices, Xi) = simulatedata(Udiff, Π, T, N, true);
opt = Opt(:LN_COBYLA, 3)
opt.xtol_rel = 1e-4
opt.max_objective = MLEObjFunc
@time (maxf,maxx,ret) = optimize(opt, [1; 1; 1])
hcat(vcat(β, δ[2]), maxx)
