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