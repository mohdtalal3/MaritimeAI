import torch

import torch.nn as nn

class RouteLSTM(nn.Module):
    def __init__(self, port_vocab_size, ship_vocab_size, hidden_dim=128):
        super().__init__()
        self.port_emb = nn.Embedding(port_vocab_size, 32)
        self.ship_emb = nn.Embedding(ship_vocab_size, 8)
        self.lstm = nn.LSTM(input_size=72, hidden_size=hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, port_vocab_size)

    def forward(self, ship_type, first_port, last_port, target_len):
        # Create embeddings
        ship_embed = self.ship_emb(ship_type).unsqueeze(1).expand(-1, target_len, -1)  # [B, T, 8]
        first_embed = self.port_emb(first_port).unsqueeze(1).expand(-1, target_len, -1)
        last_embed = self.port_emb(last_port).unsqueeze(1).expand(-1, target_len, -1)

        combined = torch.cat([first_embed, last_embed, ship_embed], dim=2)
        out, _ = self.lstm(combined)
        out = self.fc(out)
        return out



checkpoint = torch.load("best_route_lstm_model.pth")

# Extract encoders
port_encoder = checkpoint["port_encoder"]
ship_encoder = checkpoint["ship_encoder"]

# Now get vocab sizes
port_vocab_size = len(port_encoder.classes_)
ship_vocab_size = len(ship_encoder.classes_)

# Rebuild and load model
model = RouteLSTM(port_vocab_size, ship_vocab_size)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def predict_route(model, ship_type_str, first_port_str, last_port_str, max_len=10):
    model.eval()
    with torch.no_grad():
        ship_type = torch.tensor([ship_encoder.transform([ship_type_str])[0]])
        first_port = torch.tensor([port_encoder.transform([first_port_str])[0]])
        last_port = torch.tensor([port_encoder.transform([last_port_str])[0]])

        output = model(ship_type, first_port, last_port, target_len=max_len)
        pred_ids = torch.argmax(output, dim=2).squeeze(0).tolist()
        return [port_encoder.inverse_transform([i])[0] for i in pred_ids if i < len(port_encoder.classes_)]

def mapping(seq):
    seen = set()
    result = []
    for port in seq:
        if port not in seen:
            seen.add(port)
            result.append(port)
    return result

route = predict_route(model, "Cargo", "Aalborg", "Havdrup")
cleaned_route = mapping(route)
print("Predicted route:", cleaned_route)
