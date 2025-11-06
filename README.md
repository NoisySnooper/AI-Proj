Raman Spectrum Prediction Model 

This project is a complete deep learning system designed to predict a mineral's 500-point Raman spectrum ("chemical fingerprint") using only 15 chemical and physical features (its "recipe").

The system is built on a "data-first" principle, requiring a minimum of 1,000 real-world samples from the RRUFF database to function effectively.   

Core Components
The project is segmented into three primary Python files, each with a distinct responsibility :   

comprehensive_rruff_scraper_v3.py (The "Librarian")

Purpose: A robust data collection pipeline.

Features:

Downloads and processes thousands of samples from the RRUFF public database.   

Checks a local manual_rruff_data/ folder first, allowing you to manually add RRUFF data archives to save time.   

Applies strict quality filtering (e.g., must be in the 50-4000 cm⁻¹ range).   

Standardizes all spectra to a 500-point vector (200-1200 cm⁻¹) and normalizes them.   

Extracts and encodes the 15-feature input vector (10 chemical, 5 physical).   

Output: Generates the rruff_complete_dataset/ folder containing the clean rruff_features.npy and rruff_spectra.npy files required for training.   

modern_raman_models_v3.py (The "Brains")

Purpose: Contains the blueprints for four different, competing AI architectures.   

Models:

ConvNeXt1D (Primary): A state-of-the-art 1D Convolutional Neural Network (CNN) architecture, adapted from modern vision research.   

SpectraFormer (Transformer): A Transformer-based model that uses self-attention to find long-range relationships between spectrum peaks.   

CNN-LSTM Hybrid: A model combining a CNN (for local feature extraction) with a Bidirectional LSTM (for sequential pattern recognition).   

Ensemble (RF+NN): A robust hybrid model that combines a Random Forest (for stability) with a Neural Network (for complex patterns).   

advanced_training_system_v3_part1.py (The "Teacher")

Purpose: A training and evaluation harness.   

Features:

Fast Mode vs. Extended Mode: Run quick 50-epoch tests (fast_mode=True) or full 200-epoch training runs (fast_mode=False).   

Best Practices: Implements a professional training loop with AdamW, gradient clipping, ReduceLROnPlateau scheduler, and early stopping.   

Advanced Metrics: Evaluates models not just on R² score, but also on scientific relevance, including "Shape Correlation" and "Peak Position Accuracy" (within a ±20 cm⁻¹ tolerance).   

Reproducibility: Uses a fixed seed=42 for all data splits (70% train, 15% validation, 15% test).   

Installation
This project requires several common data science and deep learning libraries.

Bash
pip install torch numpy pandas scikit-learn matplotlib scipy tqdm requests beautifulsoup
Quick Start Guide
Follow these three phases to get the system running.   

Phase 1: Data Collection
This is the most critical step. The models cannot be trained without data.

(Optional but Recommended): Create a folder named manual_rruff_data in the project root. Go to the RRUFF website, manually download the "Raman" and "Chemistry" .zip archives, and place them in this folder. The scraper will find and use these local files first, which is much faster.   

Run the scraper. This script will either process your local archives or download the data automatically if the manual folder is empty.

Bash
python comprehensive_rruff_scraper_v3.py
Upon completion, you should have a new folder: rruff_complete_dataset/ containing the .npy files.

⚠️ CRITICAL DATA REQUIREMENT ⚠️
The scraper must successfully process at least 1,000 valid samples. The summary documentation explicitly warns: "Below this, all models will fail". If your scraper run produces fewer than 1,000 samples, the AI will not have enough data to learn the complex patterns.   

Phase 2: Training
Once you have data, you can train the models. The file advanced_training_system_v3_part1.py contains the RamanTrainingSystem class to handle this.   

You can run the training process from a separate Python script (e.g., main.py) or a Jupyter Notebook.

Python
# main_training.py
from advanced_training_system_v3_part1 import RamanTrainingSystem
from modern_raman_models_v3 import ConvNeXt1DModel # Import the model you want to test

# --- 1. Initialize the Training System ---
# fast_mode=True is for a quick 50-epoch test.
# fast_mode=False is for the full 200-epoch run (Target R² > 0.7)
trainer = RamanTrainingSystem(fast_mode=True)

# --- 2. Load and Prepare Data ---
features, spectra = trainer.load_data()

if features is not None:
    # Create the 70/15/15 splits
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(features, spectra)
    
    # Create PyTorch DataLoaders
    train_loader, val_loader = trainer.create_dataloaders(X_train, y_train, X_val, y_val)

    # --- 3. Initialize and Train a Model ---
    print("Initializing ConvNeXt1D Model...")
    # This project is designed to test multiple models.
    # We will start with the primary one: ConvNeXt1D
    model = ConvNeXt1DModel(input_features=X_train.shape[1], 
                            spectrum_points=y_train.shape[1])
    
    # Start the training process
    trained_model, train_losses, val_losses = trainer.train_pytorch_model(
        model=model,
        model_name="ConvNeXt1D",
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # --- 4. Evaluate the Final Model ---
    # This evaluation runs on the "final exam" (test set)
    # which the model has NEVER seen before.
    print("\n" + "="*60)
    print("PERFORMING FINAL EVALUATION ON TEST SET")
    print("="*60)
    results = trainer.evaluate_model(
        model_or_ensemble=trained_model,
        model_name="ConvNeXt1D",
        X_test=X_test,
        y_test=y_test,
        is_ensemble=False
    )
    
    print(f"\nFinal Test R² Score: {results['r2']:.4f}")
(Note: The full system is designed to loop through all four models and compare them, but the code above shows the process for training a single, primary model).   

Phase 3: GUI (Future Work)
The project is designed to feed its best-trained model into a web-based GUI for real-time predictions. The GUI components are not yet built.   

Once gui/index.html is complete, it can be opened in any browser to use the trained model.   

Project Structure
Bash
project_root/
│
├── comprehensive_rruff_scraper_v3.py   # (✔ Complete) The data scraper
├── modern_raman_models_v3.py           # (✔ Complete) The 4 model architectures
├── advanced_training_system_v3_part1.py# (Δ Needs completion) The training/eval system
├── run_complete_system_v3.py           # (X Not started) Master controller script
│
├── manual_rruff_data/                  # (User-created) Place manual.zip downloads here
│   └── *.zip
│
├── rruff_complete_dataset/             # (Generated) Output of the scraper
│   ├── rruff_features.npy
│   ├── rruff_spectra.npy
│   └── comprehensive_rruff_dataset.csv
│
├── gui/                                # (X Not started) The web-based GUI
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── best_convnext1d_model.pth           # (Generated) Output of training
├── best_spectraformer_model.pth        # (Generated) Output of training
└── model_performance_results.json      # (Generated) Output of evaluation
