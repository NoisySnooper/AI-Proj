
model_code_part2 = '''
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
    
    print("\\n   ✅ All PyTorch models working!")
    print("   ✅ Ensemble model requires fit() before predict()")

if __name__ == "__main__":
    main()
'''

# Combine parts
full_model_code = model_code_part1 + model_code_part2

with open('modern_raman_models_v3.py', 'w', encoding='utf-8') as f:
    f.write(full_model_code)

print("✅ PHASE 2 COMPLETE")
print()
print("Created: modern_raman_models_v3.py")
print("   1. ConvNeXt1D - 4 depthwise blocks, GELU, expected R² 0.65-0.80")
print("   2. SpectraFormer - 2 transformer blocks, 8-head attention, expected R² 0.55-0.70")
print("   3. CNN-LSTM - Bidirectional LSTM, expected R² 0.50-0.65")
print("   4. Ensemble - RF(100 trees) + NN, expected R² 0.55-0.70")
print()
print("All models:")
print("   - Dual output (spectrum + confidence)")
print("   - Softplus final activation")
print("   - BatchNorm + Dropout regularization")
print("   - No synthetic data generation")
print()
print("="*80)
