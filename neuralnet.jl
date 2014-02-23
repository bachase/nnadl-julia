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
	a = sigmoid( w * a .+ b )
  end
  a
end

function backpropagate(n::NeuralNet.Network, input, output)
  # a[l] are the activation values
  a = cell(size(n.bias,1)+1)
  # z[l] are the sigmoid inputs
  z = cell(size(n.bias,1))
  
  num_examples = size(input,2)

  a[1] = input
  # forward propagate
  for idx=1:length(z)
    z[idx] = n.weights[idx] * a[idx] .+ n.bias[idx]
    a[idx+1] = sigmoid( z[idx] )
  end

  # backward propogate
  dWeights = cell(size(n.weights,1))
  dBias = cell(size(n.bias,1))

     
  dBias[end] = (a[end] .- output) .* dsigmoid(z[end])
  dWeights[end] = dBias[end] * a[end-1]'
  for idx=size(dBias,1)-1:-1:1
    dBias[idx] = (n.weights[idx+1]' * dBias[idx+1]) .* dsigmoid(z[idx])
    dWeights[idx] = dBias[idx] * a[idx]'
  end
  dBias = map(m -> 1/num_examples * sum(m,2), dBias)
  dWeights = map(m -> 1/num_examples * m, dWeights)
  (dBias,dWeights)
end

function train(n::Network, data, outputs, batch_size, epochs, eta, callback)
   # train the network via backpropagation on training set defined
   # by data and labels
   # data is num_features * num_samples
   # outputs num_outputs  * num_samples
   
   num_samples = size(data,2)
   @assert size(data,2) == size(outputs,2)
   idx = [ x for x in 1:num_samples ]
   batch_offsets = vcat([x for x in 1:batch_size:num_samples],[num_samples+1])
   for epoch in 1:epochs
       @printf("Epoch %d ",epoch)
       shuffle!(idx)
       for bid in 1:length(batch_offsets)-1
           batch_ids = idx[batch_offsets[bid]:(batch_offsets[bid+1]-1)]
	       (dBias, dWeights) = backpropagate(n, data[:,batch_ids], outputs[:,batch_ids])
           for layer in 1:length(n.layers)-1
               n.weights[layer] = n.weights[layer] - eta * dWeights[layer]
               n.bias[layer] = n.bias[layer] - eta * squeeze(dBias[layer],2)
           end
       end
       callback(epoch, n)
   end
   n
end

end
