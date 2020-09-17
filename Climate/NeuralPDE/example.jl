using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, IterTools

u0 = Float32(1.1)
tspan = (0.0f0,25.0f0)

ann = FastChain(FastDense(2,16,tanh), FastDense(16,16,tanh), FastDense(16,1))
p1 = initial_params(ann)
p2 = Float32[0.5,-0.5]
p3 = [p1;p2]
θ = Float32[u0;p3]

function dudt_(du,u,p,t)
    x, y = u
    du[1] = ann(u,p[1:length(p1)])[1]
    du[2] = p[end-1]*y + p[end]*x
end
prob = ODEProblem(dudt_,u0,tspan,p3)
concrete_solve(prob,Tsit5(),[0f0,u0],p3,abstol=1e-8,reltol=1e-6)

function predict_adjoint(θ)
  Array(concrete_solve(prob,Tsit5(),[0f0,θ[1]],θ[2:end],saveat=0.0:1:25.0))
end
function loss_adjoint(θ, X, Y)
	print("X", X)
	print("Y", Y)
	sum(abs2,predict_adjoint(θ)[2,:].-1)
end
#l = loss_adjoint(θ, "x0", "y0")

cb = function (θ,l)
  println(l)
  #display(plot(solve(remake(prob,p=Flux.data(p3),u0=Flux.data(u0)),Tsit5(),saveat=0.1),ylim=(0,6)))
  return false
end

# Display the ODE with the current parameter values.
#cb(θ,l)

function mini_batch(X, Y)
	i = rand(1:size(X)[1])
	return (x[i,:], Y[i,:])
end


x = [1 2; 3 4; 5 6; 7 8]
y = [1 ; 0 ; 1; 0]
data = [mini_batch(x,y) for i in 1:20]
#train_loader = Flux.Data.Data
#loss1 = loss_adjoint(θ)
res = DiffEqFlux.sciml_train(loss_adjoint, θ, BFGS(initial_stepnorm=0.01), data, cb = cb)
