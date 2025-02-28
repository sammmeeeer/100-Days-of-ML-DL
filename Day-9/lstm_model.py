import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 


np.random.seed(0)
torch.manual_seed(0)

# Synthetic sine wave data 
t = np.linspace(0, 100, 1000)
data = np.sin(t)

# Function to create sequences 
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data, seq_length)

trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)

# LSTM Model 
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim 
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)

        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim,  batch_size, self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn 

model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


num_epochs = 100 
batch_size = trainX.size(0)
h0, c0 = None, None 

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # forward pass 
    outputs, h0, c0 = model(trainX, h0, c0)

    # compute loss 
    loss = criterion(outputs, trainY)

    # backward pass and optimize 
    loss.backward()
    optimizer.step()

    # detach hidden and cell states 
    h0 = h0.detach()
    c0 = c0.detach()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Predicted outputs 
model.eval()
predicted, _, _ = model(trainX, h0, c0)

# Adjusting the original data and prediction for plotting 
original = data[seq_length:]
time_steps = np.arange(seq_length, len(data))

plt.figure(figsize=(12, 6))
plt.plot(time_steps, original, label='Original Data')
plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
plt.title('LSTM Model Predictions vs Original Data')
plt.xlabel("Time Step")
plt.ylabel('Value')
plt.legend()
plt.show()
