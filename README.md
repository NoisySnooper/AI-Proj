# Raman Spectrum Prediction Model

A deep learning system that predicts a mineral's 500-point Raman spectrum ("chemical fingerprint") using 15 chemical and physical feature inputs.

Installation
Requires Python and the following libraries:

    pip install torch numpy pandas scikit-learn matplotlib scipy tqdm requests beautifulsoup

Usage
1. Data Collection (Critical)
You must generate the dataset before training. The scraper processes local files or downloads from RRUFF.

Option A (Fastest):

Create a folder manual_rruff_data/ in the project root.

Download "Raman" and "Chemistry" .zip files from the RRUFF website.

Place zips in the folder.

Option B (Automatic): Simply run the script (will download automatically if local files are missing): python comprehensive_rruff_scraper_v3.py

2. Training
Run the training system. You can select specific models or run the full suite.


# Training Mode: Fast vs. Full
    fast_mode=True (50 epochs), fast_mode=False (200 epochs)
    trainer = RamanTrainingSystem(fast_mode=True)
    features, spectra = trainer.load_data()

# Prepare data splits (70/15/15)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features, spectra)
    train_loader, val_loader = trainer.create_dataloaders(X_train, y_train, X_val, y_val)

# Initialize Model
    model = ConvNeXt1DModel(input_features=X_train.shape[1], spectrum_points=y_train.shape[1])

# Train
    trained_model, train_losses, val_losses = trainer.train_pytorch_model(
        model=model,
        model_name="ConvNeXt1D",
        train_loader=train_loader,
        val_loader=val_loader

# Evaluator

    trainer.evaluate_model(trained_model, "ConvNeXt1D", X_test, y_test, is_ensemble=False)

Components: comprehensive_rruff_scraper_v3.py

Role: Data Pipeline

- Scrapes/loads RRUFF data.

- Filters for quality (range 50-4000 cm⁻¹).

- Standardizes spectra (500 points, 200-1200 cm⁻¹).

- Encodes 15-feature input vector.

Components: modern_raman_models_v3.py

Role: Model Architectures

Models:

- ConvNeXt1D: (Primary) 1D CNN adapted from vision research.

- SpectraFormer: Transformer-based, uses self-attention for peak relationships.

- CNN-LSTM: Hybrid for local features and sequential patterns.

- Ensemble: Random Forest + Neural Network.

Components: advanced_training_system_v3_part1.py

Role: Training Harness

- Implements AdamW, gradient clipping, and ReduceLROnPlateau.

Metrics: R², Shape Correlation, Peak Position Accuracy (±20 cm⁻¹).

Reproducibility: Fixed seed (42).
