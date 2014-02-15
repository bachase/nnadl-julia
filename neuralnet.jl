module NeuralNet

export Network, feedforward,backpropagate
	
type Network
	# array of layer sizes, layers[1] is the size of the input layer
	layers::Array{Int64,1}
	# weights[i] is the weights connecting layer i to i+1
	# so has dimension layers[i+1] x layers[i]
	weights::Array{Array{Float64,2},1}
	# bias[i] is the output bias for neurons in layer i
	bias::Array{Array{Float64,1},1}
	
	function Network(layers::Array{Int64,1})
		bias = [randn(s) for s in layers[2:end] ]
		weights = [ randn(b,a) for (a,b) in zip(layers[1:end-1], layers[2:end]) ]
		new(layers, weights, bias)
	end
end

function sigmoid(z)
    1./(1 + exp(-z))
end

function dsigmoid(z)
    sigmoid(z) .* (1-sigmoid(z))
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

function backpropagate(n::NeuralNet.Network, input, output)
  # a[l] are the activation values
  a = cell(size(n.bias,1)+1)
  # z[l] are the sigmoid inputs
  z = cell(size(n.bias,1))
  
  a[1] = input
  # forward propagate
  for idx=1:length(z)
    z[idx] = n.weights[idx] * a[idx] + n.bias[idx]
    a[idx+1] = sigmoid( z[idx] )
  end
  # backward propogate
  dWeights = cell(size(n.weights,1))
  dBias = cell(size(n.bias,1))
  
  dBias[end] = (a[end] - output) .* dsigmoid(z[end])
  dWeights[end] = dBias[end] * a[end-1]'
  for idx=size(dBias,1)-1:-1:1
    dBias[idx] = (n.weights[idx+1]' * dBias[idx+1]) .* dsigmoid(z[idx])
    dWeights[idx] = dBias[idx] * a[idx]'
  end
  (dBias,dWeights)
end

function train(n::Network, data, labels)
   # train the network via backpropagation on training set defined
   # by data and labels
   
   #1 repeat until done
   #2 select random subset
   #3 run back-propagation
end

end
