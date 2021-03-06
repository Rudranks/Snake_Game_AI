import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Liner_QNet(nn.Module):
    # Model is a simple feet forward neural net with an input layer, a hidden layer and an output layer
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # prediction function
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name = 'model.pth'):
        # saves the path to the disk
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Pytorch optimizer: Adam optimizer for gradient descent
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Loss Fn: it is a mean squared error
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Converting all the variables to pytorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # if teh above has multiple values, it is in the form of (n, x)

        if len(state.shape) == 1:
            # if this is true then we have state in the form on only 1 number so we need to convert into (1, x) format
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # Converts done to a tuple

        # 1. predicted Q values with current state:
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Q_new = r + (y * max(next predicted q val)) -> only do if not done
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # pred.clone()
        # preds[argmax(action)] = Q_new do this to get 3 values in the next predicted q val

        self.optimizer.zero_grad()  # Empties gradient
        loss = self.criterion(target, pred)  # loss func
        loss.backward()  # back propagation and updating of gradients

        self.optimizer.step()