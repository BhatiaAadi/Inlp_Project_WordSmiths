class ImprovedTinyBERTStockPredictor(nn.Module):
    """
    Improved stock prediction model with LSTM/GRU layers instead of feed-forward
    """
    def __init__(self, input_dim=312, hidden_dim=256, lstm_layers=2, output_dim=30, lora_r=8, lora_alpha=16):
        super().__init__()

        # Add normalization layer
        self.norm_layer = NormalizationLayer(method='minmax')

        # Load TinyBERT model
        self.bert_config = AutoConfig.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.bert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

        # Apply LoRA to TinyBERT
        self.bert = LoRATransformerWrapper(self.bert, r=lora_r, alpha=lora_alpha)

        # Replace feed-forward layers with recurrent layers
        bert_output_dim = self.bert_config.hidden_size  # 312
        
        # GRU layer (could also use LSTM)
        self.gru = nn.GRU(
            input_size=bert_output_dim, 
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        # Attention mechanism for time series
        self.attention = nn.Linear(hidden_dim, 1)

        self.ffnn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        