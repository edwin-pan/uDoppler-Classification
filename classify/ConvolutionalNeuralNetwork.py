import numpy as np
import torch

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network
        
        Args:
            lrate: The learning rate for the model.
            loss_fn: A loss function defined in the following way:
            in_size: Dimension of input
            out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()

        # Convolutional Layers
        self.CNN_layer1 = torch.nn.Sequential(torch.nn.Conv2d(1,6,(5,72),stride=1, padding=2),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.MaxPool2d(2,2))

        self.CNN_layer2 = torch.nn.Sequential(torch.nn.Conv2d(6,16,(5,72),stride=1, padding=2),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.MaxPool2d(2,2))
        
        self.model = torch.nn.Sequential(torch.nn.Linear(1200, 512),
                                        torch.nn.BatchNorm1d(512),
                                        torch.nn.Dropout(0.25),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(512, 256),
                                        torch.nn.BatchNorm1d(256),
                                        torch.nn.Dropout(0.25),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Linear(256, out_size),
                                        torch.nn.LeakyReLU())

        # Learning rate
        self.lrate = lrate

        # Defined loss function
        self.loss_fn = loss_fn

        # Define optim for various gradient decent functions
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lrate, momentum=0.9)
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=lrate)

        #device = torch.device("cuda")
        #model = CNN().to(device)

    def set_parameters(self, params):
        """ Set the parameters of your network
        Args:
            params: a list of tensors containing all parameters of the network
        """
        print("set_parameters called...?")
        self.parameters = params


    def get_parameters(self):
        """ Get the parameters of your network
        Return:
            params: a list of tensors containing all parameters of the network
        """
        return list(self.model.parameters())


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        # Check input x is the correct shape
        if len(x.shape) != 4:
            x = x.view((-1,1,200,72))

        # Apply CNN
        yy = self.CNN_layer1(x)
        # yy = self.CNN_layer2(yy)

        # Apply NN
        y_ = yy.view((yy.shape[0],-1)) # can also call torch.flatten(x, 1) -> this allows for "reshapping" inside sequential
        y_p = self.model(y_)
        return y_p # torch.ones(x.shape[0], 1)


    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        
        # Clear Optimizer gradient
        self.optimizer.zero_grad()

        # Evaluate x to get y_p
        y_p = self.forward(x)
        
        # Optional L2 Regulation
        L2_reg = 0
        for param in self.model.parameters():
            L2_reg += torch.norm(param, 2)**2

        # Compare y to y_p
        lr_reg = 0.0005

        loss = self.loss_fn(y_p,y) + lr_reg*L2_reg
        
        # Calculate Error, perform backpropogation
        self.model.zero_grad() # Unessessary
        loss.backward()

        # Update gradients
        self.optimizer.step()

        return np.float(loss)


def batcher(n_items, batch_size, seed=0, debug=False, debug_iter=0):
    """Return a batch of values to be used
    
    **debug flag** makes batch return deterministic
    
    Returns indices for set and label
    """
    if debug:
        return np.arange(debug_iter*100,(debug_iter+1)*100)
    else:
        rnd_function = np.random.RandomState(seed)
        possible_indices = np.arange(n_items)
        rnd_function.shuffle(possible_indices)
        return possible_indices[0:batch_size]


def Normalize(data, sample=True):
    if sample:
        mean = torch.mean(data,1, True)
        std = torch.std(data, axis=1, keepdims=True)
    else:
        mean = torch.mean(data)
        std = torch.std(data)
    centered = data-mean
    scaled = centered/std
    return scaled


def fit(train_set, train_labels, test_set, n_iter, batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    Args:
        train_set: an (N, out_size) torch tensor
        train_labels: an (N,) torch tensor
        test_set: an (M,) torch tensor
        n_iter: int, the number of epochs of training
        batch_size: The size of each batch to train on. (default 100)

    Returns:
        losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
        yhats: an (M,) NumPy array of approximations to labels for test_set
        net: A NeuralNet object

    """
    in_size=0
    print("Start training")

    # train_set = torch.tensor(train_set,dtype=torch.float32)
    train_labels = torch.tensor(train_labels,dtype=torch.int64)
    # test_set = torch.tensor(test_set,dtype=torch.float32)

    # train_NP = train_set.numpy()
    # train_TO = torch.from_numpy(train_NP)

    # Force Seed
    torch.manual_seed(0)

    out_size = 2
    num_img = train_set.shape[0]

    loss_fn = torch.nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = NeuralNet(0.03, loss_fn, in_size, out_size).to_device()

    net = NeuralNet(0.03, loss_fn, in_size, out_size).to(device)

    train_set = Normalize(train_set,sample=False)
    test_set = Normalize(test_set,sample=False)
    
    # Epochs 
    epochs = 6

    # Get batches
    num_batches = num_img//batch_size

    # Reshape to -1,1,200,72
    train_set = train_set.view((-1,1,200,72))
    test_set = test_set.view((-1,1,200,72))

    # Training
    for k in range(epochs):
        print("Epoch: ", k)
        losses = np.zeros((n_iter))
        net.train()
        for i in range(num_batches):
            # idx = batcher(num_img, batch_size, debug=True,debug_iter=i)
            idx = batcher(num_img, batch_size, seed=int(np.random.rand()*100))
            
            curr_set = train_set[idx].to(device)
            curr_lab = train_labels[idx].to(device)
            
            for j in range(n_iter):
                losses[j] = net.step(curr_set, curr_lab)
                # print("[#"+str(i)+"] iter:"+str(j)+" Loss:"+str(losses[j]))

            if i % 5 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (i + 1, j + 1, losses[j]))

    torch.save(net,"net.model")

    # # Evaluate
    # net.eval()
    # yhats_raw = net(test_set).detach().numpy()
    # yhats = [sample[0]<sample[1] for sample in yhats_raw]
    

    print("[LOSS]: ", min(losses),max(losses))
    # return losses, yhats, net
    return losses, net
