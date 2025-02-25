import numpy as np
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
    inputs = np.array([
        ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
        ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
        ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
        ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
        ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
    ])
    expected = np.array([
        ["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"],
        ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
        ["C","E","G","I","K","M","O","Q","S","U","W","Y","A","B","D","F","H","J","L","N","P","R","T","V","X","Z"],
        ["N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L","M"],
        ["I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H"]
    ])
    
    # Encode strings to int indexes
    input_encoded = np.vectorize(string.ascii_uppercase.index)(inputs)
    expected_encoded = np.vectorize(string.ascii_uppercase.index)(expected)
    
    # Convert to PyTorch tensors
    # PyTorch RNN expects input shape as (seq_len, batch, input_size)
    # We'll transpose the data to match this format
    X = torch.FloatTensor(input_encoded).long()
    y = torch.FloatTensor(expected_encoded).long()
    
    # One-hot encode inputs and targets
    # PyTorch has a cleaner way to handle this with embedding or using CrossEntropyLoss
    X_one_hot = torch.nn.functional.one_hot(X, num_classes=26).float()
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=26).float()
    
    # Reshape to (batch_size, seq_len, features)
    X_one_hot = X_one_hot
    y_one_hot = y_one_hot
    
    # Create PyTorch model
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleRNN, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
            rnn_out, _ = self.rnn(x)
            # rnn_out shape: (batch_size, seq_len, hidden_size)
            output = self.fc(rnn_out)
            # output shape: (batch_size, seq_len, output_size)
            return output
    
    # Initialize model, loss function, and optimizer
    model = SimpleRNN(input_size=26, hidden_size=128, output_size=26)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_one_hot)
        
        # Compute loss
        # Reshape for CrossEntropyLoss: [batch_size*seq_len, num_classes]
        loss = criterion(
            outputs.reshape(-1, 26), 
            y.reshape(-1)
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Test with new input
    new_inputs = np.array([["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"]])
    new_inputs_encoded = np.vectorize(string.ascii_uppercase.index)(new_inputs)
    new_inputs_tensor = torch.tensor(new_inputs_encoded).long()
    new_inputs_one_hot = torch.nn.functional.one_hot(new_inputs_tensor, num_classes=26).float()
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(new_inputs_one_hot)
        
    # Get prediction of last time step and last element
    prediction = torch.argmax(prediction[0, -1]).item()
    
    print(prediction)
    print(string.ascii_uppercase[prediction])
