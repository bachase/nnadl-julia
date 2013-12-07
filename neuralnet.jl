module NeuralNet

export Network, feedforward
	
type Network
	# array of layer sizes, layers[1] is the size of the input layer
	layers::Array{Int64,1}
	# weights[i] is the weights connecting layer i to i+1
	# so has dimension layers[i+1] x layers[i]
	weights::Array{Array{Float64,2},1}
	# bias[i] is the output bias for neurons in layer i
	bias::Array{Array{Float64,2},1}
	
	function Network(layers::Array{Int64,1})
		bias = [randn(s,1) for s in layers[2:end] ]
		weights = [ randn(b,a) for (a,b) in zip(layers[1:end-1], layers[2:end]) ]
		new(layers, weights, bias)
	end
end

function sigmoid(z)
	1./(1 + exp(-z))
end

function feedforward(n::Network, input)
  # run the input through the neural network
  # i.e., if a[1] = input then iterate
  #   a[i+1] = sigmoid(w[i] * a[i] + b[i])
  a = input
  for (w,b) in zip(n.weights, n.bias)
	a = sigmoid( w * a + b )
  end
  a
end

end