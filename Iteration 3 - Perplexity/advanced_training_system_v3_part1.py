# advanced_training_system_v3.py
# Complete training pipeline per comprehensive prompt
# Fast/Extended modes, automatic model comparison, advanced metrics

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from modern_raman_models_v3 import (
    ConvNeXt1DModel, SpectraFormerModel, CNNLSTMModel, EnsembleModel
)

class RamanTrainingSystem:
    """
    Complete training system implementing comprehensive prompt specifications.

    Features:
    - Fast (50 epochs) vs Extended (200 epochs) modes
    - 70/15/15 train/val/test split with seed=42 for reproducibility
    - AdamW optimizer with ReduceLROnPlateau
    - Loss: 0.7 MSE + 0.3 L1 + 0.01 confidence
    - Gradient clipping, early stopping
    - Automatic model comparison
    - Advanced metrics per spec
    """

    def __init__(self, data_path="./rruff_complete_dataset", fast_mode=True):
        """
        Initialize training system.

        Args:
            data_path: Path to processed RRUFF data
            fast_mode: True for Fast (50 epochs), False for Extended (200 epochs)
        """
        self.data_path = Path(data_path)
        self.fast_mode = fast_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training parameters per spec
        if fast_mode:
            self.epochs = 50
            self.batch_size = 32
            self.lr = 1e-3
            self.patience = 20
            print("⚡ FAST MODE selected")
            print(f"   Epochs: {self.epochs}")
            print(f"   Batch size: {self.batch_size}")
            print(f"   Learning rate: {self.lr}")
            print(f"   Expected time: 2-3 hours (CPU) or 30-60 min (GPU)")
            print(f"   Target: R² > 0.5")
        else:
            self.epochs = 200
            self.batch_size = 16
            self.lr = 5e-4
            self.patience = 30
            print("🚀 EXTENDED MODE selected")
            print(f"   Epochs: {self.epochs}")
            print(f"   Batch size: {self.batch_size}")
            print(f"   Learning rate: {self.lr}")
            print(f"   Expected time: 12-24 hours (CPU) or 2-4 hours (GPU)")
            print(f"   Target: R² > 0.7")

        print(f"\n🖥️  Using device: {self.device}")

    def load_data(self):
        """
        Load processed RRUFF data.
        Per spec: Check for 1,000+ samples minimum.
        """
        print("\n📂 Loading RRUFF dataset...")

        try:
            features = np.load(self.data_path / "rruff_features.npy")
            spectra = np.load(self.data_path / "rruff_spectra.npy")

            print(f"   ✅ Loaded:")
            print(f"      Samples: {len(features)}")
            print(f"      Features: {features.shape[1]} dimensions")
            print(f"      Spectra: {spectra.shape[1]} points")

            # Check minimum per spec
            if len(features) < 1000:
                print(f"\n   ⚠️  WARNING: Only {len(features)} samples!")
                print("   Spec recommends 1,000+ minimum for acceptable results")
                print("   3,000+ for target performance")
                response = input("\n   Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return None, None

            # Data quality checks
            print(f"\n   📊 Data quality:")
            print(f"      Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"      Spectra range: [{spectra.min():.3f}, {spectra.max():.3f}]")
            print(f"      Missing values: {np.isnan(features).sum() + np.isnan(spectra).sum()}")

            # Check normalization per spec
            if spectra.min() < 0 or spectra.max() > 1.5:
                print("   ⚠️  Spectra may not be properly normalized to [0,1]")

            return features, spectra

        except FileNotFoundError:
            print(f"   ❌ Data files not found in {self.data_path}")
            print("   Run comprehensive_rruff_scraper_v3.py first!")
            return None, None

    def prepare_data(self, features, spectra):
        """
        Prepare train/val/test splits.
        Per spec: 70/15/15 split with seed=42 for reproducibility.
        """
        print("\n🔧 Preparing data splits...")

        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, spectra,
            test_size=0.30,  # 30% for val+test
            random_state=42,
            shuffle=True
        )

        # Second split: val vs test (equal sizes)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,  # Half of 30% = 15% each
            random_state=42,
            shuffle=True
        )

        print(f"   ✅ Splits created:")
        print(f"      Training:   {len(X_train):5d} samples ({len(X_train)/len(features)*100:.1f}%)")
        print(f"      Validation: {len(X_val):5d} samples ({len(X_val)/len(features)*100:.1f}%)")
        print(f"      Testing:    {len(X_test):5d} samples ({len(X_test)/len(features)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataloaders(self, X_train, y_train, X_val, y_val):
        """Create PyTorch DataLoaders"""
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return train_loader, val_loader

    def compute_loss(self, spectrum_pred, spectrum_true, confidence):
        """
        Compute loss per spec: 0.7 MSE + 0.3 L1 + 0.01 confidence calibration.

        Args:
            spectrum_pred: (batch, 500) predicted spectra
            spectrum_true: (batch, 500) true spectra
            confidence: (batch,) confidence scores

        Returns:
            total_loss, reconstruction_loss, confidence_loss
        """
        # Reconstruction: 70% MSE + 30% L1 per spec
        mse_loss = F.mse_loss(spectrum_pred, spectrum_true)
        l1_loss = F.l1_loss(spectrum_pred, spectrum_true)
        reconstruction_loss = 0.7 * mse_loss + 0.3 * l1_loss

        # Confidence calibration per spec
        # Target: confidence should match quality (low error = high confidence)
        per_sample_error = torch.mean((spectrum_pred - spectrum_true) ** 2, dim=1)
        target_confidence = torch.exp(-per_sample_error * 5.0)  # Scale factor
        confidence_loss = F.mse_loss(confidence, target_confidence)

        # Total loss per spec
        total_loss = reconstruction_loss + 0.01 * confidence_loss

        return total_loss, reconstruction_loss, confidence_loss

    def train_pytorch_model(self, model, model_name, train_loader, val_loader):
        """
        Train individual PyTorch model.
        Per spec: AdamW, ReduceLROnPlateau, gradient clipping, early stopping.

        Returns:
            trained model, train_losses, val_losses
        """
        print(f"\n🔥 Training {model_name}...")
        print("="*60)

        model = model.to(self.device)
        model.train()

        # Optimizer per spec
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5  # Per spec
        )

        # Scheduler per spec
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=10 if self.fast_mode else 15,
            factor=0.5
        )

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward
                spectrum_pred, confidence = model(batch_X)

                # Loss
                total_loss, recon_loss, conf_loss = self.compute_loss(
                    spectrum_pred, batch_y, confidence
                )

                # Backward
                total_loss.backward()

                # Gradient clipping per spec
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_train_loss += total_loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            epoch_val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    spectrum_pred, confidence = model(batch_X)

                    total_loss, _, _ = self.compute_loss(
                        spectrum_pred, batch_y, confidence
                    )

                    epoch_val_loss += total_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Scheduler step
            scheduler.step(avg_val_loss)

            # Progress reporting (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train: {avg_train_loss:.6f} | "
                      f"Val: {avg_val_loss:.6f} | "
                      f"LR: {current_lr:.6f}")

            # Early stopping per spec
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                          f'best_{model_name.lower().replace(" ", "_")}_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\n   ⏹️  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(
            torch.load(f'best_{model_name.lower().replace(" ", "_")}_model.pth')
        )

        print(f"   ✅ {model_name} training complete!")
        print(f"      Best val loss: {best_val_loss:.6f}")

        return model, train_losses, val_losses

    def evaluate_model(self, model_or_ensemble, model_name, X_test, y_test, is_ensemble=False):
        """
        Comprehensive model evaluation per spec.

        Metrics:
        - R² Score (primary)
        - MSE, MAE
        - Peak position accuracy
        - Shape correlation

        Returns:
            metrics dict
        """
        print(f"\n🔍 Evaluating {model_name}...")

        if is_ensemble:
            # Ensemble predict
            predictions, confidence = model_or_ensemble.predict(X_test)
        else:
            # PyTorch model predict
            model_or_ensemble.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(self.device)
                predictions, confidence = model_or_ensemble(X_tensor)
                predictions = predictions.cpu().numpy()
                confidence = confidence.cpu().numpy()

        # Standard metrics per spec
        r2 = r2_score(y_test, predictions, multioutput='uniform_average')
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        # Advanced metrics per spec

        # 1. Shape correlation
        correlations = []
        for i in range(min(100, len(y_test))):
            corr = np.corrcoef(y_test[i], predictions[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        shape_correlation = np.mean(correlations)

        # 2. Peak position accuracy per spec
        peak_accuracies = []
        for i in range(min(50, len(y_test))):
            # Find peaks (prominence=0.1 to avoid noise)
            true_peaks, _ = find_peaks(y_test[i], prominence=0.1)
            pred_peaks, _ = find_peaks(predictions[i], prominence=0.1)

            if len(true_peaks) == 0 and len(pred_peaks) == 0:
                peak_accuracies.append(1.0)
            elif len(true_peaks) == 0 or len(pred_peaks) == 0:
                peak_accuracies.append(0.0)
            else:
                # Count matches within ±20 points (per spec: ±20 cm⁻¹)
                matches = 0
                for true_pk in true_peaks:
                    distances = np.abs(pred_peaks - true_pk)
                    if np.min(distances) <= 20:
                        matches += 1
                accuracy = matches / max(len(true_peaks), len(pred_peaks))
                peak_accuracies.append(accuracy)

        peak_accuracy = np.mean(peak_accuracies) if peak_accuracies else 0.0

        # Results
        print(f"   Metrics:")
        print(f"      R² Score:           {r2:.6f}")
        print(f"      MSE:                {mse:.6f}")
        print(f"      MAE:                {mae:.6f}")
        print(f"      Shape Correlation:  {shape_correlation:.3f}")
        print(f"      Peak Accuracy:      {peak_accuracy:.3f}")
        print(f"      Avg Confidence:     {np.mean(confidence):.3f} ± {np.std(confidence):.3f}")

        # Performance assessment per spec
        if r2 > 0.7:
            status = "🎉 EXCELLENT - Target exceeded!"
        elif r2 > 0.5:
            status = "✅ GOOD - Minimum target met"
        elif r2 > 0.3:
            status = "⚠️  FAIR - Below target"
        else:
            status = "❌ POOR - Needs improvement"

        print(f"\n   Overall: {status}")

        return {
            'r2': float(r2),
            'mse': float(mse),
            'mae': float(mae),
            'shape_correlation': float(shape_correlation),
            'peak_accuracy': float(peak_accuracy),
            'avg_confidence': float(np.mean(confidence)),
            'std_confidence': float(np.std(confidence)),
            'predictions': predictions,
            'confidence': confidence
        }

    def run_full_pipeline(self):
        """
        Execute the full data loading, training, and evaluation pipeline.
        """
        # Load data
        features, spectra = self.load_data()
        if features is None:
            return

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(features, spectra)
        train_loader, val_loader = self.create_dataloaders(X_train, y_train, X_val, y_val)

        # Models to train
        models_to_train = {
            "ConvNeXt1D": ConvNeXt1DModel(input_features=X_train.shape[1]),
            "SpectraFormer": SpectraFormerModel(input_features=X_train.shape[1]),
            "CNNLSTM": CNNLSTMModel(input_features=X_train.shape[1]),
        }

        trained_models = {}
        all_results = {}

        # Train and evaluate each model
        for name, model in models_to_train.items():
            trained_model, _, _ = self.train_pytorch_model(model, name, train_loader, val_loader)
            trained_models[name] = trained_model
            
            results = self.evaluate_model(trained_model, name, X_test, y_test)
            all_results[name] = {k: v for k, v in results.items() if k not in ['predictions', 'confidence']}

        # Train and evaluate ensemble
        ensemble = EnsembleModel(
            list(trained_models.values()),
            input_features=X_train.shape[1]
        )
        ensemble.fit(X_val, y_val) # Fit meta-learner on validation set
        
        ensemble_results = self.evaluate_model(ensemble, "Ensemble", X_test, y_test, is_ensemble=True)
        all_results["Ensemble"] = {k: v for k, v in ensemble_results.items() if k not in ['predictions', 'confidence']}

        # Final summary
        print("\n\n" + "="*60)
        print("🏆 FINAL RESULTS SUMMARY")
        print("="*60)
        
        # Create a DataFrame for easy comparison
        results_df = pd.DataFrame(all_results).T
        results_df = results_df.sort_values(by='r2', ascending=False)
        print(results_df[['r2', 'mse', 'shape_correlation', 'peak_accuracy']])

        # Save results to JSON
        with open("training_results.json", "w") as f:
            json.dump(all_results, f, indent=4)
        print("\n   💾 Full results saved to training_results.json")


if __name__ == "__main__":
    # To run, create an instance of the system and call the pipeline
    # Example: Fast mode
    training_system = RamanTrainingSystem(fast_mode=True)
    training_system.run_full_pipeline()

    # Example: Extended mode (for best results)
    # training_system = RamanTrainingSystem(fast_mode=False)
    # training_system.run_full_pipeline()
