import torch
from torch import nn

class QuestionPairMLP(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_size, hidden_size_1, hidden_size_2, device):
        super(QuestionPairMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32, device=device))
        self.embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(embedding_size * 2, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, 2)
    def forward(self, x1, x2):
        q1_embedded = self.embedding(x1).mean(dim=1)
        q2_embedded = self.embedding(x2).mean(dim=1)
        # Assumes x1 and x2 are averages of the embedding for all the tokens in the q1 and q2 sequences
        # x = torch.abs(q1_embedded - q2_embedded)
        x = torch.cat((q1_embedded, q2_embedded), dim=1)
        hidden_1 = self.relu1(self.fc1(x))
        hidden_2 = self.relu2(self.fc2(hidden_1))
        output = self.fc3(hidden_2)
        # We don't need softmax since the CrossEntropyLoss will apply it anyway
        return output

class QuestionPairCosineSimilarity(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_size, device):
        super(QuestionPairCosineSimilarity, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32, device=device))
        self.embedding.weight.requires_grad = False
        self.cos = nn.CosineSimilarity()
        # Binary classification
        self.fc = nn.Linear(1, 2)
    def forward(self, x1, x2):
        q1_embedded = self.embedding(x1).mean(dim=1)
        q2_embedded = self.embedding(x2).mean(dim=1)
        cos_output = self.cos(q1_embedded, q2_embedded).unsqueeze(1)
        return self.fc(cos_output)

class QuestionPairLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_size, hidden_size, num_layers, device):
        super(QuestionPairLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32, device=device))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x1, x2):
        q1_embedded = self.embedding(x1)
        q2_embedded = self.embedding(x2)
        
        # Get the LSTM outputs
        q1_lstm_out, _ = self.lstm(q1_embedded)
        q2_lstm_out, _ = self.lstm(q2_embedded)
        
        # We use the final states of the LSTM (both directions)
        q1_lstm_out = torch.cat((q1_lstm_out[:, -1, :self.lstm.hidden_size], q1_lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        q2_lstm_out = torch.cat((q2_lstm_out[:, -1, :self.lstm.hidden_size], q2_lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        
        # Concatenate the outputs from both questions
        x = torch.cat((q1_lstm_out, q2_lstm_out), dim=1)
        
        # Pass through fully connected layers
        hidden = self.relu(self.fc1(x))
        output = self.fc2(hidden)
        
        return output
