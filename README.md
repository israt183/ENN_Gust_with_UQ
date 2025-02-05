# Wind Gust Prediction with Uncertainty Qunatification with Evidential Neural Network

This repository contains selected scripts used in our research on **wind gust prediction and uncertainty quantification** using Evidential Neural Networks (ENN). The scripts demonstrate **hyperparameter tuning, model training, spatial gradient analysis, active learning**, and **visualization of uncertainty** in wind gust predictions.  

⚠️ **Data Availability:** Due to research agreements, the dataset used in this study cannot be shared at this time. The dataset will be made available after publication. However, the scripts can be formatted to be used with a different dataset, provided the necessary adjustments are made.  

📄 **Preprint:** The preprint related to this work is available at [arXiv](https://doi.org/10.48550/arXiv.2502.00300).  

---

The repository includes the following scripts for reference purposes:

### **1️⃣ Hyperparameter Tuning**
- **Script:** `Hyperparameter_tuning.py`
- **Description:** Performs hyperparameter tuning using **Optuna** with the **ECHO** Python package.
- **Dependency:** [ECHO (NCAR)](https://github.com/NCAR/echo-opt)

### **2️⃣ Model Training & Testing**
- **Script:** `Model_train_and_test.ipynb`
- **Description:** Trains an **Evidential Neural Network (ENN)** and performs inference on test data.
- **Dependency:** [MILES-GUESS](https://github.com/ai2es/miles-guess/tree/main/mlguess)

### **3️⃣ Spatial Wind Gust Gradient Analysis**
- **Script:** `Spatial_gradient.ipynb`
- **Description:** Computes **spatial gradients of predicted wind gusts** for the study domain.

### **4️⃣ Wind Gust Prediction & Uncertainty Visualization**
- **Script:** `Spatial_plot.ipynb`
- **Description:** Generates spatial maps of **wind gust predictions and their uncertainty** over the **Northeast USA**.

### **5️⃣ Active Learning with ENN**
- **Script:** `Active_learning_ENN_data_parallelized.ipynb`
- **Description:** Implements **active learning** using the **Evidential Neural Network**.

### **6️⃣ Maximum Surface Wind Speed Tracking (NCL)**
- **Script:** `storm_track_WS_sea_masked.ncl`
- **Description:** Computes **maximum surface wind speeds** and their locations at each hour over a storm duration.
- **Note:** This script requires **WRF output files** from the storm simulation.

---

## **🔧 Usage**
These scripts can be adapted to work with other datasets by modifying the data preprocessing steps. Ensure that your dataset follows a similar structure to the one used in this research. 

## **📜 Citation**
If you find this repository useful, please cite our preprint :
@misc{jahan2025uncertaintyquantificationwindgust,
      title={Uncertainty Quantification of Wind Gust Predictions in the Northeast US: An Evidential Neural Network and Explainable Artificial Intelligence Approach}, 
      author={Israt Jahan and John S. Schreck and David John Gagne and Charlie Becker and Marina Astitha},
      year={2025},
      eprint={2502.00300},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.00300}, 
}

### **🔹 Setting Up the Environment**
To run the Python scripts, install the required dependencies:
```bash
pip install -r requirements.txt
