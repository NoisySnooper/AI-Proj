# modern_raman_models_v3.py
# Four competing architectures per comprehensive prompt
# ConvNeXt1D, SpectraFormer, CNN-LSTM, Ensemble

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MODEL 1: ConvNeXt1D (Primary Recommendation)
# Expected R²: 0.65-0.80
# =============================================================================

class ConvNeXtBlock1D(nn.Module):
    """
    ConvNeXt block adapted for 1D spectral data.
    Per spec: depthwise separable convolutions, GELU activation.
    """
    def __init__(self, dim, drop_path=0.0):
        super().__init__()

        # Depthwise convolution (groups=dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise convolutions with expansion
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()  # Per spec
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Optional drop path for regularization
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        """
        x: (batch, channels, length)
        """
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Permute for layer norm: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Pointwise MLPs
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Permute back: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # Residual connection
        x = residual + self.drop_path(x)

        return x

class ConvNeXt1DModel(nn.Module):
    """
    ConvNeXt1D for Raman spectrum prediction.
    Per spec: Encoder → ConvNeXt blocks → Decoder → 500 points
    Expected R²: 0.65-0.80 (best performer)
    """
    def __init__(self, input_features=15, spectrum_points=500):
        super().__init__()

        self.spectrum_points = spectrum_points

        # Feature encoder: 15 → 512
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.GELU(),
            nn.BatchNorm1d(512)
        )

        # Project to sequence: 512 → (64 channels × 64 length)
        self.to_sequence = nn.Linear(512, 64 * 64)

        # ConvNeXt blocks
        self.convnext_blocks = nn.ModuleList([
            ConvNeXtBlock1D(64, drop_path=0.05),
            ConvNeXtBlock1D(64, drop_path=0.05),
            ConvNeXtBlock1D(64, drop_path=0.05),
            ConvNeXtBlock1D(64, drop_path=0.05)
        ])

        # Decoder: upsample to 500 points
        self.decoder = nn.Sequential(
            # 64 → 256 length via transposed conv
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(32),

            # 256 → 512 length
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(16),

            # Final projection
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )

        # Adaptive pooling to exactly 500 points
        self.final_pool = nn.AdaptiveAvgPool1d(spectrum_points)

        # Confidence head (dual output per spec)
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Confidence in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 15) feature vectors

        Returns:
            spectrum: (batch, 500) predicted spectra
            confidence: (batch,) confidence scores
        """
        # Encode features
        features = self.feature_encoder(x)  # (B, 512)

        # Project to sequence
        seq = self.to_sequence(features)  # (B, 4096)
        seq = seq.view(-1, 64, 64)  # (B, 64, 64)

        # Apply ConvNeXt blocks
        for block in self.convnext_blocks:
            seq = block(seq)

        # Get confidence before upsampling
        confidence = self.confidence_head(seq).squeeze(-1)  # (B,)

        # Decode to spectrum
        spectrum = self.decoder(seq)  # (B, 1, ~512)
        spectrum = spectrum.squeeze(1)  # (B, ~512)
        spectrum = self.final_pool(spectrum.unsqueeze(1)).squeeze(1)  # (B, 500)

        # Softplus to ensure positive (per spec)
        spectrum = F.softplus(spectrum)

        return spectrum, confidence

# =============================================================================
# MODEL 2: SpectraFormer (Transformer)
# Expected R²: 0.55-0.70
# =============================================================================

class MultiHeadSpectralAttention(nn.Module):
    """
    Multi-head attention for spectral relationships.
    Per spec: good for interpretability.
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        B, L, D = x.size()

        # Linear projections and reshape for multi-head
        Q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B, H, L, d_k)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        # Final linear
        output = self.w_o(attended)

        return output

class SpectraFormerModel(nn.Module):
    """
    Transformer for Raman spectrum prediction.
    Per spec: Attention-based, multi-head, positional encoding.
    Expected R²: 0.55-0.70
    """
    def __init__(self, input_features=15, spectrum_points=500, d_model=256, n_heads=8):
        super().__init__()

        self.spectrum_points = spectrum_points
        self.d_model = d_model

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_model)
        )

        # Learnable positional encoding for spectrum positions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, spectrum_points, d_model) * 0.02
        )

        # Transformer layers
        self.attention1 = MultiHeadSpectralAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.attention2 = MultiHeadSpectralAttention(d_model, n_heads)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm4 = nn.LayerNorm(d_model)

        # Output heads
        self.spectrum_head = nn.Linear(d_model, 1)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 15) feature vectors

        Returns:
            spectrum: (batch, 500)
            confidence: (batch,)
        """
        B = x.size(0)

        # Encode input features
        features = self.feature_encoder(x)  # (B, d_model)

        # Expand to sequence with positional encoding
        # Broadcast features across sequence length
        sequence = features.unsqueeze(1).expand(-1, self.spectrum_points, -1)  # (B, 500, d_model)
        sequence = sequence + self.pos_encoding  # Add positional info

        # Transformer block 1
        attn_out = self.attention1(sequence)
        sequence = self.norm1(sequence + attn_out)
        ffn_out = self.ffn1(sequence)
        sequence = self.norm2(sequence + ffn_out)

        # Transformer block 2
        attn_out = self.attention2(sequence)
        sequence = self.norm3(sequence + attn_out)
        ffn_out = self.ffn2(sequence)
        sequence = self.norm4(sequence + ffn_out)

        # Generate spectrum
        spectrum = self.spectrum_head(sequence).squeeze(-1)  # (B, 500)
        spectrum = F.softplus(spectrum)  # Positive values per spec

        # Generate confidence (average over sequence)
        pooled = torch.mean(sequence, dim=1)  # (B, d_model)
        confidence = self.confidence_head(pooled).squeeze(-1)  # (B,)

        return spectrum, confidence

# =============================================================================
# MODEL 3: CNN-LSTM Hybrid
# Expected R²: 0.50-0.65
# =============================================================================

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM for Raman prediction.
    Per spec: CNN for local features, LSTM for sequential patterns.
    Expected R²: 0.50-0.65
    """
    def __init__(self, input_features=15, spectrum_points=500):
        super().__init__()

        self.spectrum_points = spectrum_points

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.GELU(),
            nn.BatchNorm1d(512)
        )

        # Project to sequence: 512 → (128 channels × 32 length)
        self.to_sequence = nn.Linear(512, 128 * 32)

        # CNN layers for local feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        # Bidirectional LSTM per spec
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        # Decoder: upsample to 500 points
        # Bidirectional LSTM outputs 128 channels (64*2)
        self.decoder = nn.Sequential(
            # 32 → 128 length
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(64),

            # 128 → 256 length
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(32),

            # 256 → 512 length
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            # Final
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )

        # Adaptive pooling to exactly 500 points
        self.final_pool = nn.AdaptiveAvgPool1d(spectrum_points)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 64),  # 128 from bidirectional hidden
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 15)

        Returns:
            spectrum: (batch, 500)
            confidence: (batch,)
        """
        # Encode features
        features = self.feature_encoder(x)  # (B, 512)

        # Project to sequence
        seq = self.to_sequence(features)  # (B, 4096)
        seq = seq.view(-1, 128, 32)  # (B, 128, 32)

        # CNN processing
        cnn_out = self.cnn_layers(seq)  # (B, 32, 32)

        # LSTM processing (needs B, L, C)
        lstm_input = cnn_out.permute(0, 2, 1)  # (B, 32, 32)
        lstm_out, (hidden, cell) = self.lstm(lstm_input)  # (B, 32, 128)

        # Confidence from final hidden state (concat forward + backward)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, 128)
        confidence = self.confidence_head(final_hidden).squeeze(-1)

        # Decode to spectrum (back to B, C, L)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, 128, 32)
        spectrum = self.decoder(lstm_out)  # (B, 1, ~512)
        spectrum = spectrum.squeeze(1)  # (B, ~512)
        spectrum = self.final_pool(spectrum.unsqueeze(1)).squeeze(1)  # (B, 500)

        # Softplus per spec
        spectrum = F.softplus(spectrum)

        return spectrum, confidence

# =============================================================================
# MODEL 4: Ensemble (Random Forest + Neural Network)
# Expected R²: 0.55-0.70
# =============================================================================

class EnsembleModel:
    """
    Ensemble of Random Forest + Neural Network.
    Per spec: RF for robustness, NN for non-linearity.
    Expected R²: 0.55-0.70
    """
    def __init__(self, input_features=15, spectrum_points=500):
        self.spectrum_points = spectrum_points
        self.input_features = input_features

        # Random Forest: one regressor per output point (expensive but thorough)
        # Per spec: 100 trees per point
        print("      Initializing Random Forest ensemble (100 trees × 500 points)...")
        self.rf_models = [
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42 + i,
                n_jobs=-1
            )
            for i in range(spectrum_points)
        ]

        # Neural network component
        self.nn_model = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),

            nn.Linear(2048, spectrum_points),
            nn.Softplus()  # Per spec
        )

        # Confidence model
        self.confidence_model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move NN to device
        self.nn_model = self.nn_model.to(self.device)
        self.confidence_model = self.confidence_model.to(self.device)

    def fit(self, X, y, epochs=50, lr=0.001, verbose=True):
        """
        Train both RF and NN components.

        Args:
            X: (N, 15) features
            y: (N, 500) spectra
            epochs: training epochs for NN
            lr: learning rate
        """
        if verbose:
            print("      Training Random Forest...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train RF ensemble
        for i, rf in enumerate(tqdm(self.rf_models, desc="      RF Progress", 
                                   disable=not verbose)):
            rf.fit(X_scaled, y[:, i])

        if verbose:
            print("      Training Neural Network...")

        # Train NN
        optimizer = torch.optim.AdamW(
            list(self.nn_model.parameters()) + list(self.confidence_model.parameters()),
            lr=lr,
            weight_decay=1e-5
        )

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.nn_model.train()
        self.confidence_model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward
            nn_pred = self.nn_model(X_tensor)
            confidence = self.confidence_model(X_tensor).squeeze(-1)

            # Loss: 70% MSE + 30% L1 per spec
            mse_loss = F.mse_loss(nn_pred, y_tensor)
            l1_loss = F.l1_loss(nn_pred, y_tensor)
            loss = 0.7 * mse_loss + 0.3 * l1_loss

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.nn_model.parameters()) + list(self.confidence_model.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

        if verbose:
            print("      ✅ Ensemble training complete")

    def predict(self, X):
        """
        Make ensemble predictions.
        Per spec: 30% RF + 70% NN weighting.

        Returns:
            spectra: (N, 500)
            confidence: (N,)
        """
        X_scaled = self.scaler.transform(X)

        # RF predictions
        rf_predictions = np.array([
            rf.predict(X_scaled) for rf in self.rf_models
        ]).T  # (N, 500)

        # NN predictions
        self.nn_model.eval()
        self.confidence_model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            nn_predictions = self.nn_model(X_tensor).cpu().numpy()
            confidence = self.confidence_model(X_tensor).cpu().numpy().squeeze()

        # Ensemble weighting per spec
        ensemble_pred = 0.3 * rf_predictions + 0.7 * nn_predictions

        return ensemble_pred, confidence

    def to(self, device):
        """Move NN components to device"""
        self.device = device
        self.nn_model = self.nn_model.to(device)
        self.confidence_model = self.confidence_model.to(device)
        return self

def main():
    """Test model instantiation"""
    print("🧠 Testing model architectures...")

    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 15)

    # Test each model
    models = {
        'ConvNeXt1D': ConvNeXt1DModel(),
        'SpectraFormer': SpectraFormerModel(),
        'CNN-LSTM': CNNLSTMModel()
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            spectrum, confidence = model(x)
        print(f"   ✅ {name}: spectrum {spectrum.shape}, confidence {confidence.shape}")

    print("\n   ✅ All PyTorch models working!")
    print("   ✅ Ensemble model requires fit() before predict()")

if __name__ == "__main__":
    main()
