import torch.nn as nn
import torch.optim as optim
import torch

class PairEventClassifier(nn.Module):
    def __init__(self, input_dim, pos_weight=0.5, hidden_dim=832):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
        )

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])) 
        # SGD = stochastic gradient descent
        # TODO: Adam optimizer func, try different learning rates
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


    def forward(self, x):
        return self.model(x).squeeze(-1)
    
    
    def fit(self, trainloader, validationloader=None):
        # training example from docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

        self.model.train()
        prev_loss = 0.0
        for epoch in range(20):  # loop over the dataset multiple times

            running_loss = 0.0
            for data in trainloader:
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            if validationloader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for data in validationloader:
                        inputs, labels = data
                        outputs = self.forward(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        val_batches += 1

                val_loss = val_loss / val_batches if val_batches else 0.0
                print(f'Epoch {epoch + 1:3d} loss: {running_loss:.3f} val_loss: {val_loss:.3f}')
                if (running_loss > prev_loss - 0.01 and running_loss < prev_loss + 0.01):
                    break
                prev_loss = running_loss
                
                self.model.train()
            else:
                print(f'Epoch {epoch + 1:3d} loss: {running_loss:.3f}')


    def test(self, testloader):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        self.model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                # predict outputs by running through the model
                predicted = self.model(inputs)
                # yes this is applied elementwise
                # predicted = torch.round(predicted.squeeze())
                predicted = (predicted > 0.0).int().squeeze()

                true_positive += ((predicted == 1) & (labels == 1)).sum()
                true_negative += ((predicted == 0) & (labels == 0)).sum()
                false_positive += ((predicted == 1) & (labels == 0)).sum()
                false_negative += ((predicted == 0) & (labels == 1)).sum()

        total = true_positive+true_negative+false_negative+false_positive
        print('Confusion matrix:')
        print(f'TP: {true_positive}, FP: {false_positive}')
        print(f'FN: {false_negative}, TN: {true_negative}')
        print(f'Overall accuracy: {100 * (true_negative + true_positive) / total:.3f}%')

    
    def save(self, filepath):
        # TODO: save model
        # i think...
        # torch.save(self.model.state_dict(), filepath)
        pass
