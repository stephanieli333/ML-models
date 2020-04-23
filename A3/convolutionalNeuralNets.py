import numpy as np
import sklearn as sk
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Importing the modules we need above

# This is the class that is called in STEP 2
class Net(nn.Module):
    def __init__(self):
        # This is our super consructor, within our class' constructor
        super(Net, self).__init__()

        #NOTE: Order is conv1 --> Pool --> conv2 --> fc1 --> fc2 --> fc3
        #Important to see that our activation size goes down as we go deeper into the neural net
        # Modifying our neural networks to take 3-channel images in and spit 6 out
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # The second step takes 6-channel images in and spits 16 out
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Ensuring the dimensions are combinations/multiplications of conv1 and conv2
        # Specifying the dimensions of the fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Applying the rectified linear unit function element-wise.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Ensuring the dimensions are the same as the ones in the fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        # Once again, applying the rectified linear unit function element-wise on fc1 and fc2
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to show the image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # Calling matplotlib to show the image
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #####plt.show()

# Helper function!!
def accuracyPredictor(images,testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            # Predicting the one where the max energy is matched
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # Adding to our correct predictions if labels match predictions
            correct += (predicted == labels).sum().item()

    # Printing out accuracy of the network
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    accuracyResult = 100*correct/total
    return accuracyResult

if __name__ == "__main__":

    ########## STEP 1: Loading and normalizing the data set CIFAR 10 ##########

    # Here we are normalizing with the standard mean, std, etc. of 0.5
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Creating our train set and test set, and loading via torch utilities and torchvision
    # Built in methods allow us to directly load the train and test sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    # Method parameters are binary so they allow shuffling + direct access to train/test sets
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # The classes that will be called later to help aid our picture classification task
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Here, as the guide suggests, we show some random training images.
    # This code follows till the end of Step 1

    # Get some random training images.
    # Created using an iterator
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Showing the images
    imshow(torchvision.utils.make_grid(images))
    #  Printing the labels out
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########## STEP 2: Define a Convolutional Neural Network ##########

    # Calling the class Net here.
    # Comments explaining this class and its parameters are in the respective class and its methods
    net = Net()

    ########## STEP 3: Define a Loss function and optimizer ##########

    # Using Classification Cross-Entropy loss
    # Furthermore, we use Stochastic Gradient Descent with momentum to optimize it.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########## STEP 4: Train the network ##########
    epochResults = []
    noOfEpochs = []
    for int in range(2,7):
        # We loop over our data iterator, and feed the inputs to the network and optimize.
        yAxisAccuracy = []
        xAxisNumEpochs = []
        f = 1
        for epoch in range(int):  # loop over the dataset multiple times
            yAxisAccuracy = []
            xAxisNumEpochs = []
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # Calling the optimizer from step 3
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))

                    yAxisAccuracy.append((running_loss/2000))
                    xAxisNumEpochs.append(i+1)

                    #plt.xlabel('Num of batches - Epoch: ' + str(f))


                    running_loss = 0.0

            #plt.plot(xAxisNumEpochs, yAxisAccuracy)
            #plt.xlabel('Num of batches - Epoch: ' + str(f))
            #plt.ylabel('Loss')
            ##### plt.show()
            f += 1

        print('Finished Training')

        # Making sure that our trained model is saved via torch
        #PATH = './cifar_net.pth'
        #torch.save(net.state_dict(), PATH)

        ########## STEP 5: Test the network on the test data ##########

        # Loading in a form that is iterable
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        # print images
        # Ensuring that the class labels are also there with the "Ground Truth" (aka actual names)
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

        # Loading our net.
        # This is not really necessary since we wrote it above, but good practice since we might have
        # had to load it in at some point
        #net = Net()
        #net.load_state_dict(torch.load(PATH))


        outputs = net(images)

        # The higher the energy for a class, the more the network thinks that the image is of the particular class.
        # Therefore below we are predicting the energies and trying to identify the "highest" one
        _, predicted = torch.max(outputs, 1)

        # Printing out predictions
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(4)))



        # This is evaluating our networks performance on the whole dataset
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                # Predicting the one where the max energy is matched
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # Adding to our correct predictions if labels match predictions
                correct += (predicted == labels).sum().item()

        # Printing out accuracy of the network
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

        epochResults.append(100*correct/total)
        noOfEpochs.append(int)


        outputs = 0
        print("Result of Epoch: ", epochResults)
        print("No of Epochs: ", int)



        # Below we analyze the classes that did well and the classes that did not
        # We make sure to iterate through all the correct and all the total classes
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                # Squeezing if our labels are same as predicted
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # Printing out the accuracy of the best classes
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

        print("\n END OF ONE EPOCH ITERATION: " + str(int))
        print("\n")


    ##### TRAINING ON GPU ######

    plt.clf() # Clears the entire figure
    plt.plot(noOfEpochs,epochResults)
    plt.xlabel('No. Of Training Epochs')
    plt.ylabel('Average Accuracy')
    plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device available: ", device)


    net.to(device)

    inputs, labels = data[0].to(device), data[1].to(device)

