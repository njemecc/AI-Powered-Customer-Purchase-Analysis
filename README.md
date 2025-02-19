# AI-Powered Customer Purchase Analysis

This project demonstrates an end-to-end AI solution for analyzing retail customer purchase data and providing product recommendations. It includes:

- **Data Generation** (synthetic CSV creation)  
- **Data Analysis** (top products/categories, spending statistics)  
- **Customer Classification** (K-Means)  
- **Hybrid Recommendation System** (Time-Weighted Collaborative + Content-Based)  
- **PDF Report Generation** (final summary of findings)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing Functionalities](#testing-functionalities)
---

## Project Structure


1. **data/**  
   - Stores generated CSV and PDF outputs after running the code.
2. **main.py**  
   - Main Python script that executes the entire pipeline: data generation, analysis, classification, recommendation, and report creation.
3. **documentation.pdf** (optional)  
   - Contains a more detailed explanation of the methodology, diagrams, or any additional analysis.
4. **requirements.txt**  
   - Lists Python libraries required for running this project.

---

## Installation

1. **Clone** this repository or download the ZIP.

2. **Install dependencies**
     ``` bash
         pip install -r requirements.txt
      ```
     
3. **Make sure you have Python installed**


## Usage

1. **Run the main script**:
   ```bash
      python main.py
   ```

2. **This will**:
   - Generate a synthetic dataset: `data/synthetic_purchase_data.csv`  
   - Perform data analysis (e.g., top categories, products, spending stats)  
   - Classify customers using K-Means  
   - Provide a hybrid recommendation example  
   - Produce a PDF report: `data/customer_analysis_report.pdf`

3. **Review Outputs**  
   - Open `customer_analysis_report.pdf` in the `data/` folder to see:
     - Top product categories and items  
     - Sample spending statistics  
     - Customer segmentation (Low, Medium, High Spenders)  
     - An example recommendation for one selected customer



## Testing Functionalities

- **CSV Generation**  
  Confirm that `data/synthetic_purchase_data.csv` is created with approximately 5,000 purchase records.

- **PDF Report**  
  Verify that `data/customer_analysis_report.pdf` is generated and includes all relevant sections (e.g., data analysis, customer segmentation, and recommendations).

- **Parameter Tuning**  
  In `main.py`, modify `weight_collab`, `weight_content`, or `alpha` to see how different weighting and time-decay settings affect the recommendation process.  
  Rerun `python main.py` to observe and test these changes.


