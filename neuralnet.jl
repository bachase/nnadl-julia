module NeuralNet

export Network, feedforward, backpropagate, cost

using NumericExtensions
	
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

function sigmoid(z::Real)
    one(z)/(one(z) + exp(-z))
end
@vectorize_1arg Real sigmoid

function dsigmoid(z::Real)
    exp(-z)/((one(z)+exp(-z))*(one(z)+exp(-z)))
end
@vectorize_1arg Real dsigmoid

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

function sum_weights_sq(n::Network)
    res = 0
    for wt in n.weights
        res += sum(wt.^2)
    end
    res
end

function cost(n::Network, input, output, lambda = 0)
    # standard neural network cost function with optional weight regularization lambda
    noutput = NeuralNet.feedforward(n,input)
    err = (noutput - output).^2
    1./size(input,2) * 0.5 * sum(err) + lambda/2.0 * sum_weights_sq(n)
end



function backpropagate(n::NeuralNet.Network, input, output)
  # Run back-propagation to calculate derivatives of square-error cost without regularization term 
  #  n is the network
  #  input is num_features * num_examples
  #  output is num_outpust * num_examples
 
  num_examples = size(input,2)
  @assert size(input,2) == size(output,2)
  
  ## Run forward prop and store results
  # a[l] are the activation values
  a = cell(size(n.bias,1)+1)
  # z[l] are the sigmoid inputs
  z = cell(size(n.bias,1))

  a[1] = input
  for idx=1:length(z)
    z[idx] = n.weights[idx] * a[idx] .+ n.bias[idx]
    a[idx+1] = sigmoid( z[idx] )
  end

  ## Backpropogate
  dWeights = cell(size(n.weights,1))
  dBias = cell(size(n.bias,1))
     
  dBias[end] = (a[end] .- output) .* dsigmoid(z[end])
  dWeights[end] = dBias[end] * a[end-1]'
  for idx=size(dBias,1)-1:-1:1
    dBias[idx] = (n.weights[idx+1]' * dBias[idx+1]) .* dsigmoid(z[idx])
    dWeights[idx] = dBias[idx] * a[idx]'
  end
  
  # normalize and sum out examples dimension
  dBias = map(m -> 1/num_examples * sum(m,2), dBias)
  dWeights = map(m -> 1/num_examples * m, dWeights)
  (dBias,dWeights)
end

function dummy_callback(epoch, n::Network)
# do nothing
end

function train(n::Network, data, outputs, epochs = 10, batch_size = 1, eta = 0.95, lambda = 0, callback = dummy_callback)
   # Train the network via stochastic gradient descent on training set defined
   # by data and labels
   #
   #  data is num_features * num_samples
   #  outputs is num_outputs  * num_samples
   #  epochs is the number of iteratations to run
   #  batch_size is size of mini-batches in each epoch
   #  eta is learning rate
   #  lambda is regularization coefficient
   #  callback is called at the end of each epoch iteration with the current epoch number
   #        and the current network n
   
   num_samples = size(data,2)
   @assert size(data,2) == size(outputs,2)
   
   # For stochastic mini-batch run, we need to pick random subsets of samples
   # However, we do not "own" the data, so we should not shuffle in place.
   # We also do not want to make a copy since data may be large, so instead
   # I make an indexing vector and shuffle that
   idx = [ x for x in 1:num_samples ]
   batch_offsets = vcat([x for x in 1:batch_size:num_samples],[num_samples+1])
   
   for epoch in 1:epochs
       shuffle!(idx)
       for bid in 1:length(batch_offsets)-1
           batch_ids = idx[batch_offsets[bid]:(batch_offsets[bid+1]-1)]
	       (dBias, dWeights) = backpropagate(n, data[:,batch_ids], outputs[:,batch_ids])
           for layer in 1:length(n.layers)-1
               n.weights[layer] = n.weights[layer] - eta * ( dWeights[layer] + lambda * n.weights[layer] )
               n.bias[layer] = n.bias[layer] - eta * squeeze(dBias[layer],2)
           end
       end
       callback(epoch, n)
   end
   n
end

end
