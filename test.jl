reload("neuralnet.jl")

# <codecell>

using HDF5;
using ProfileView;

# <codecell>

fid = h5open("./data/mnist.hdf5","r")

# <codecell>

dump(fid)

function make_onehot(ids)
    res = zeros(10, length(ids))
    for (idx, id) in enumerate(ids)
        res[id+1,idx] = 1
    end
    res
end

images = read(fid["training_data/images"])
labels = make_onehot(read(fid["training_data/labels"]))
test_images = read(fid["test_data/images"])[:,1:1000]
test_labels = make_onehot(read(fid["test_data/labels"]))[:,1:1000]

# <codecell>

epochs = 40
train_err = zeros(epochs)
test_err = zeros(epochs)
function callback(epoch, net)
    train_err[epoch] = NeuralNet.cost(net, images, labels)
    test_err[epoch] = NeuralNet.cost(net, test_images, test_labels) 
end
n = NeuralNet.Network([784,30,10]);

#code_typed(NeuralNet.train, (typeof(n), typeof(images), typeof(labels), typeof(epochs), typeof(200), typeof(0.98), typeof(0)))
NeuralNet.train(n, images, labels, epochs, 50000, 0.98, 0)
Profile.clear()
@time NeuralNet.train(n, images, labels, epochs, 5000, 0.98, 0)
#Profile.print()
#ProfileView.view()
# <codecell>


