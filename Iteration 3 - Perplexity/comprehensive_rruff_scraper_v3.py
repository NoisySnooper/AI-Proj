# comprehensive_rruff_scraper_v3.py
# Complete RRUFF data collection system with local archive priority
# Built from comprehensive prompt specifications

import os
import zipfile
import io
import requests
import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveRRUFFScraper:
    """
    RRUFF data scraper implementing comprehensive prompt specifications.

    Key features:
    - Checks for local archives FIRST (manual download support)
    - Falls back to automated scraping if needed
    - Extracts 15 features: 10 composition + 5 properties
    - Quality filtering: >50 points, valid range, normalized [0,1]
    - Target: 1,000-3,000 samples minimum
    """

    def __init__(self, data_folder="./rruff_complete_dataset_v3", 
                 manual_folder="./rruff_complete_dataset"):
        """
        Initialize scraper.

        Args:
            data_folder: Output folder for processed data
            manual_folder: Folder to check for pre-downloaded archives
        """
        self.output_dir = Path(data_folder)
        self.manual_dir = Path(manual_folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "https://rruff.info"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Crystal system encoding per spec
        self.crystal_systems = {
            'cubic': 1, 'tetragonal': 2, 'orthorhombic': 3,
            'hexagonal': 4, 'trigonal': 5, 'monoclinic': 6, 'triclinic': 7
        }

        # Mineral class encoding per spec
        self.mineral_classes = {
            'silicate': 1, 'oxide': 2, 'carbonate': 3, 'sulfide': 4,
            'sulfate': 5, 'halide': 6, 'phosphate': 7, 'native': 8
        }

        print("🔧 RRUFF Scraper Initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Manual archive check: {self.manual_dir}")

    def check_existing_data(self):
        """
        Check if processed data already exists.
        Returns True if complete dataset found.
        """
        required_files = [
            self.output_dir / "rruff_features.npy",
            self.output_dir / "rruff_spectra.npy",
            self.output_dir / "comprehensive_rruff_dataset.csv"
        ]

        if all(f.exists() for f in required_files):
            # Load and check sample count
            features = np.load(required_files[0])
            if len(features) >= 1000:  # Per spec minimum
                print("✅ Existing dataset found:")
                print(f"   Samples: {len(features)}")
                print(f"   Features: {features.shape[1]}")
                print(f"   Location: {self.output_dir}")
                return True

        return False

    def load_local_archives(self):
        """
        PRIORITY: Check for and extract local .zip archives.
        Implements "Option B: Manual archives" from spec.

        Returns dict of extracted directories or None if no archives found.
        """
        print("\n📂 STEP 1: Checking for local archives...")

        if not self.manual_dir.exists():
            print(f"   ⚠️  Manual folder not found: {self.manual_dir}")
            print("   Proceeding to automated download...")
            return None

        # Find all .zip files
        archives = list(self.manual_dir.rglob("*.zip"))

        if not archives:
            print("   ⚠️  No .zip archives found in manual folder")
            print("   Proceeding to automated download...")
            return None

        print(f"   Found {len(archives)} archive(s):")
        for arc in archives:
            print(f"      - {arc.name}")

        # Extract archives
        extracted = {}

        for archive in archives:
            name_lower = archive.stem.lower()

            # Determine destination based on filename
            if any(x in name_lower for x in ['raman', 'excellent', 'oriented']):
                dest = self.output_dir / "raman_spectra"
                key = 'raman_spectra'
            elif any(x in name_lower for x in ['chemistry', 'microprobe']):
                dest = self.output_dir / "chemistry_data"
                key = 'chemistry_data'
            elif any(x in name_lower for x in ['cell', 'unit', 'parameters']):
                dest = self.output_dir / "cell_parameters"
                key = 'cell_parameters'
            else:
                print(f"   ⚠️  Skipping unknown archive: {archive.name}")
                continue

            # Extract
            dest.mkdir(parents=True, exist_ok=True)
            print(f"   📦 Extracting {archive.name} → {dest}")

            try:
                with zipfile.ZipFile(archive, 'r') as zf:
                    zf.extractall(dest)
                extracted[key] = dest
                print(f"      ✅ Extracted successfully")
            except Exception as e:
                print(f"      ❌ Extraction failed: {e}")

        if extracted:
            print(f"\n   ✅ Extracted {len(extracted)} archive type(s)")
            return extracted

        return None

    def download_bulk_data(self):
        """
        Automated download as fallback if no local archives.
        """
        print("\n📥 STEP 2: Automated bulk download from RRUFF...")

        bulk_urls = {
            'raman_spectra': 'https://www.rruff.net/zipped_data_files/raman/',
            'chemistry_data': 'https://www.rruff.net/zipped_data_files/chemistry/'
        }
        downloaded = {}

        for key, url in bulk_urls.items():
            dest = self.output_dir / key
            dest.mkdir(parents=True, exist_ok=True)

            print(f"\n   📦 Downloading {key}...")
            print(f"      URL: {url}")

            try:
                response = self.session.get(url, stream=True, timeout=300)

                if response.status_code == 200:
                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    buf = io.BytesIO()

                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"   {key}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            buf.write(chunk)
                            pbar.update(len(chunk))

                    # Extract
                    print(f"      📂 Extracting...")
                    buf.seek(0)
                    with zipfile.ZipFile(buf) as zf:
                        zf.extractall(dest)

                    downloaded[key] = dest
                    print(f"      ✅ Success: {dest}")

                else:
                    print(f"      ❌ HTTP {response.status_code}")

            except Exception as e:
                print(f"      ❌ Error: {e}")

        return downloaded if downloaded else None

    def parse_spectrum_file(self, file_path):
        """
        Parse individual Raman spectrum file.
        Per spec: 50-4000 cm⁻¹ range, remove negatives, interpolate to 500 points.

        Returns: (wavenumbers, intensities) or (None, None) if invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            wavenumbers = []
            intensities = []

            for line in lines:
                line = line.strip()

                # Skip comments and empty lines
                if line.startswith('#') or line.startswith('##') or not line:
                    continue

                # Parse data
                parts = re.split(r'[\s,;]+', line)
                if len(parts) >= 2:
                    try:
                        wn = float(parts[0])
                        intensity = float(parts[1])

                        # Quality filter per spec
                        if 50 <= wn <= 4000 and intensity >= 0:
                            wavenumbers.append(wn)
                            intensities.append(intensity)
                    except ValueError:
                        continue

            # Check minimum points (spec requires 50+)
            if len(wavenumbers) < 50:
                return None, None

            return np.array(wavenumbers), np.array(intensities)

        except Exception:
            return None, None

    def interpolate_to_standard_grid(self, wavenumbers, intensities):
        """
        Interpolate spectrum to standard 500-point grid (200-1200 cm⁻¹).
        Per spec: linear interpolation, normalize to [0,1].

        Returns: 500-point normalized intensity array
        """
        # Standard grid per spec
        standard_wn = np.linspace(200, 1200, 500)

        # Interpolate
        standard_intensity = np.interp(standard_wn, wavenumbers, intensities)

        # Normalize to [0,1] per spec
        if standard_intensity.max() > 0:
            standard_intensity = standard_intensity / standard_intensity.max()

        return standard_intensity

    def parse_chemical_formula(self, formula_str):
        """
        Parse chemical formula to extract element percentages.
        Handles common oxide formulas (SiO₂, FeO, etc.).

        Returns: dict of element percentages
        """
        composition = {
            'Mg': 0.0, 'Fe': 0.0, 'Si': 0.0, 'Al': 0.0, 'Ca': 0.0,
            'Na': 0.0, 'K': 0.0, 'Ti': 0.0, 'Mn': 0.0, 'Cr': 0.0
        }

        if not formula_str or formula_str == 'unknown':
            return composition

        # Simple regex to find elements and numbers
        # Matches patterns like Fe2SiO4, MgO, etc.
        pattern = r'([A-Z][a-z]?)([0-9.]*)'
        matches = re.findall(pattern, formula_str)

        element_counts = {}
        for element, count in matches:
            if element in composition:
                count_val = float(count) if count else 1.0
                element_counts[element] = element_counts.get(element, 0) + count_val

        # Normalize to percentages
        total = sum(element_counts.values())
        if total > 0:
            for elem in composition:
                if elem in element_counts:
                    composition[elem] = (element_counts[elem] / total) * 100

        return composition

    def extract_metadata(self, sample_id, raman_dir, chemistry_data=None):
        """
        Extract comprehensive metadata for a sample.
        Per spec: mineral name, formula, hardness, density, crystal system, class.

        Returns: metadata dict with 15-feature specification
        """
        metadata = {
            'sample_id': sample_id,
            'mineral_name': 'unknown',
            'chemical_formula': 'unknown',
            'hardness': 5.0,  # Default per spec
            'density': 3.0,   # Default per spec
            'crystal_system': 'cubic',
            'mineral_class': 'silicate',
            'space_group': 1
        }

        # Try to get chemistry data if available
        if chemistry_data and sample_id in chemistry_data:
            chem = chemistry_data[sample_id]
            metadata.update({
                'mineral_name': chem.get('mineral_name', metadata['mineral_name']),
                'chemical_formula': chem.get('formula', metadata['chemical_formula']),
                'hardness': float(chem.get('hardness', metadata['hardness'])),
                'density': float(chem.get('density', metadata['density'])),
                'crystal_system': chem.get('crystal_system', metadata['crystal_system']),
                'mineral_class': chem.get('mineral_class', metadata['mineral_class'])
            })

        return metadata

    def load_chemistry_data(self, chemistry_dir):
        """
        Load chemistry data from extracted folder.
        Parses chemistry files to dict keyed by sample_id.
        """
        print("\n🧪 Loading chemistry data...")

        if not chemistry_dir or not chemistry_dir.exists():
            print("   ⚠️  Chemistry directory not found")
            return {}

        chemistry_dict = {}

        # Look for chemistry files
        chem_files = list(chemistry_dir.rglob("*.txt")) + \
                     list(chemistry_dir.rglob("*.csv"))

        print(f"   Found {len(chem_files)} chemistry file(s)")

        for file_path in chem_files:
            try:
                # Try CSV first
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')

                # Extract relevant columns
                for _, row in df.iterrows():
                    # Find sample ID column
                    sample_id = None
                    for col in ['sample_id', 'Sample_ID', 'ID', 'RRUFF_ID']:
                        if col in row and pd.notna(row[col]):
                            sample_id = str(row[col]).strip()
                            break

                    if sample_id:
                        chemistry_dict[sample_id] = dict(row)

            except Exception as e:
                continue

        print(f"   ✅ Loaded chemistry for {len(chemistry_dict)} samples")
        return chemistry_dict

    def create_feature_vector(self, composition, metadata):
        """
        Create 15-dimensional feature vector per spec.

        Features:
        [0-9]: Mg, Fe, Si, Al, Ca, Na, K, Ti, Mn, Cr percentages
        [10]: Hardness (1-10)
        [11]: Density (g/cm³)
        [12]: Crystal system (encoded 1-7)
        [13]: Space group (default 1)
        [14]: Mineral class (encoded 1-8)
        """
        features = [
            composition['Mg'],   # 0
            composition['Fe'],   # 1
            composition['Si'],   # 2
            composition['Al'],   # 3
            composition['Ca'],   # 4
            composition['Na'],   # 5
            composition['K'],    # 6
            composition['Ti'],   # 7
            composition['Mn'],   # 8
            composition['Cr'],   # 9
            float(metadata['hardness']),  # 10
            float(metadata['density']),   # 11
            self.crystal_systems.get(metadata['crystal_system'].lower(), 1),  # 12
            int(metadata['space_group']),  # 13
            self.mineral_classes.get(metadata['mineral_class'].lower(), 1)   # 14
        ]

        return features

    def process_all_spectra(self, raman_dir, chemistry_data):
        """
        Main processing loop: parse all spectra, extract features.
        Per spec: Target 1,000-3,000 samples.
        """
        print("\n📊 STEP 3: Processing Raman spectra...")

        if not raman_dir or not raman_dir.exists():
            print("   ❌ Raman directory not found!")
            return None, None, None

        # Find all spectrum files
        spectrum_files = list(raman_dir.rglob("*.txt"))
        print(f"   Found {len(spectrum_files)} spectrum file(s)")

        if len(spectrum_files) < 100:
            print(f"   ⚠️  WARNING: Only {len(spectrum_files)} files found!")
            print("   Minimum target is 1,000+ for good results")

        all_features = []
        all_spectra = []
        all_metadata = []

        # Process with progress bar
        for file_path in tqdm(spectrum_files, desc="   Processing"):
            # Parse spectrum
            wavenumbers, intensities = self.parse_spectrum_file(file_path)

            if wavenumbers is None:
                continue

            # Interpolate to standard grid
            standard_spectrum = self.interpolate_to_standard_grid(
                wavenumbers, intensities
            )

            # Extract sample ID
            sample_id = file_path.stem.split('__')[0]

            # Get metadata
            metadata = self.extract_metadata(sample_id, raman_dir, chemistry_data)

            # Parse composition
            composition = self.parse_chemical_formula(metadata['chemical_formula'])

            # Create feature vector (15 dimensions per spec)
            features = self.create_feature_vector(composition, metadata)

            # Store
            all_features.append(features)
            all_spectra.append(standard_spectrum)
            all_metadata.append({
                'sample_id': sample_id,
                'mineral_name': metadata['mineral_name'],
                'formula': metadata['chemical_formula']
            })

        print(f"\n   ✅ Processed {len(all_features)} valid spectra")

        if len(all_features) < 1000:
            print(f"   ⚠️  WARNING: Only {len(all_features)} samples!")
            print("   Spec recommends 1,000+ for minimum quality")
            print("   3,000+ for target performance")

        return (np.array(all_features), 
                np.array(all_spectra), 
                all_metadata)

    def save_processed_data(self, features, spectra, metadata):
        """
        Save processed data in formats required by training system.
        Per spec: .npy for arrays, .csv for full dataset.
        """
        print("\n💾 STEP 4: Saving processed data...")

        # Save numpy arrays
        np.save(self.output_dir / "rruff_features.npy", features)
        np.save(self.output_dir / "rruff_spectra.npy", spectra)

        # Create comprehensive CSV
        df_meta = pd.DataFrame(metadata)

        # Add features as columns
        feature_names = ['Mg%', 'Fe%', 'Si%', 'Al%', 'Ca%', 'Na%', 'K%', 
                        'Ti%', 'Mn%', 'Cr%', 'Hardness', 'Density', 
                        'Crystal_System', 'Space_Group', 'Mineral_Class']

        for i, name in enumerate(feature_names):
            df_meta[name] = features[:, i]

        # Add spectrum as JSON string (for reference)
        df_meta['spectrum'] = [spec.tolist() for spec in spectra]

        df_meta.to_csv(self.output_dir / "comprehensive_rruff_dataset.csv", 
                      index=False)

        print("   ✅ Saved:")
        print(f"      - {self.output_dir / 'rruff_features.npy'}")
        print(f"      - {self.output_dir / 'rruff_spectra.npy'}")
        print(f"      - {self.output_dir / 'comprehensive_rruff_dataset.csv'}")
        print()
        print("   📊 Dataset Statistics:")
        print(f"      Total samples: {len(features)}")
        print(f"      Feature dimensions: {features.shape[1]}")
        print(f"      Spectrum points: {spectra.shape[1]}")
        print(f"      Feature range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"      Spectra range: [{spectra.min():.2f}, {spectra.max():.2f}]")

        return True

    def run_complete_pipeline(self):
        """
        Execute complete data collection pipeline.
        Per spec: Check existing → Local archives → Download → Process → Save
        """
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE RRUFF DATA COLLECTION PIPELINE")
        print("="*80)

        # Step 0: Check if already done
        if self.check_existing_data():
            response = input("\n   Use existing data? (y/n): ")
            if response.lower() == 'y':
                return True

        # Step 1: Try local archives first
        data_dirs = self.load_local_archives()

        # Step 2: Fall back to download if needed
        if not data_dirs:
            data_dirs = self.download_bulk_data()

        if not data_dirs:
            print("\n❌ FAILED: Could not obtain data from any source!")
            return False

        # Step 2.5: Load chemistry data
        chemistry_data = {}
        if 'chemistry_data' in data_dirs:
            chemistry_data = self.load_chemistry_data(data_dirs['chemistry_data'])

        # Step 3: Process spectra
        features, spectra, metadata = self.process_all_spectra(
            data_dirs.get('raman_spectra'),
            chemistry_data
        )

        if features is None or len(features) < 100:
            print("\n❌ FAILED: Insufficient valid spectra!")
            return False

        # Step 4: Save
        success = self.save_processed_data(features, spectra, metadata)

        if success:
            print("\n" + "="*80)
            print("🎉 DATA COLLECTION COMPLETE!")
            print("="*80)
            print()
            print("✅ Ready for model training!")
            print()
            if len(features) >= 1000:
                print("✅ Sample count meets minimum spec (1,000+)")
            else:
                print(f"⚠️  Only {len(features)} samples - may need more for best results")

        return success

def main():
    """Main execution"""
    scraper = ComprehensiveRRUFFScraper()
    success = scraper.run_complete_pipeline()

    if not success:
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Check if .zip files are in ./manual_rruff_data/")
        print("   2. Verify internet connection for automated download")
        print("   3. Ensure sufficient disk space (2-5 GB)")

        return 1

    return 0

if __name__ == "__main__":
    exit(main())
