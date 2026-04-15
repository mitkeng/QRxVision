# QRxVision
Domain-specific compound-kinase assignment

<img width="350"  alt="image" src="https://github.com/user-attachments/assets/b7bfb1fe-1753-4add-aa69-d829fd0ba2a5" />

# Compound Similarity Analyzer

## 🚀 1. Setup and Installation

To get started, you'll need the `run_qrxvision.py` script, the `requirements.txt` file, and the necessary data files.

### Clone the Repository

```bash
git clone https://github.com/mitkeng/QRxVision.git
cd your-repository-name
```

### 📂 Required Components
* 📜 **Scripts:** `run_qrxvision.py`, `requirements.txt`
* 📊 **Datasets:** `Drug_ML_RE_training.csv`, `Cancer_drug3.csv`, `abnTOTAL_dataset3.csv`
```text

├── run_qrxvision.py          # Main execution script
├── requirements.txt          # Package dependencies
├── Drug_ML_RE_training.csv   # Dataset 1
├── Cancer_drug3.csv          # Dataset 2
└── abnTOTAL_dataset3.csv     # Dataset 3

```


### Install Dependencies
Open your terminal or command prompt, navigate to the directory where you placed the files, and install the required Python libraries using `uv pip` (or `pip` if `uv` is not installed):

```bash
# Using uv (recommended for speed)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt

```

### 🛠 2. Running the run_qrxvision.py Script

The `run_qrxvision.py` script is designed to be run from the command line using `argparse` to handle different input scenarios.

**Command Structure:** The basic command structure is `python run_qrxvision.py [arguments]`.

### Available Arguments:

*   `--smile <SMILE_STRING>`: (Required for single compound processing) Provide a single SMILES string for analysis.
*   `--name <COMPOUND_NAME>`: (Optional, for single compound) Provide a name for the compound being analyzed.
*   `--csv_file <PATH_TO_CSV>`: (Required for batch processing) Provide the path to a CSV file containing compounds. This CSV must have a column named `smile` for SMILES strings and can optionally have a `name` column.
*   `--output_file <OUTPUT_PATH.csv>`: (Optional) Specify a path to save the similarity results to a CSV file. If not provided, results will only be printed to the console.
*   `--top_n <NUMBER>`: (Optional) Specify the number of top similar compounds to display/save. Defaults to 10.

### Single Compound Processing Example:
To find the top 7 similar compounds for Ripretinib and save the results:

```bash
python run_qrxvision.py --smile "CCN1C2=CC(=NC=C2C=C(C1=O)C3=CC(=C(C=C3Br)F)NC(=O)NC4=CC=CC=C4)NC" --name "Ripretinib" --output_file single_compound_results.csv --top_n 7
```

## 📊 3. Understanding the Output

### Console Output
The script will print the processing status and the top N similar compounds directly to your terminal.

### CSV Output File
If you specify `--output_file`, a CSV file will be generated. This file will contain details for each query compound (Name, SMILE, Test Image filename) and its top similar reference compounds (Reference Compound filename without .png extension, Similarity Score).

**Example structure for `single_compound_results.csv`:**


| TestCompoundName | TestCompoundSMILE | TestImage | ReferenceCompound | SimilarityScore |
| :--- | :--- | :--- | :--- | :--- |
| Ripretinib | CCN1C2=CC(...) | test_4.png | Cenerimod | 0.9729 |
| Ripretinib | CCN1C2=CC(...) | test_161.png | Erdafitinib | 0.9711 |
| ... | ... | ... | ... | ... |





