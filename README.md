# Pune Apartment Price Predictor

**Live app:** _add your Streamlit link here_ · Built with scikit-learn and Streamlit

Predicts Pune apartment prices (in lakhs) from area, BHK, bathrooms, balconies and locality.

## Run
```bash
pip install -r requirements.txt
python prep_data.py data/Pune_property_data.csv   # raw listings -> data/pune_house_data.csv
python train.py                                   # trains, saves model/
streamlit run app.py
```

## Data cleaning (prep_data.py)
37,539 raw listings -> 25,215 usable apartments: removed blank and repeated-header rows,
7,857 exact duplicates, plots/villas/houses, rows with unparseable area or missing values;
converted price from rupees to lakhs.

## Modelling (train.py)
Further outlier removal (price per sq ft within each locality, sq ft per bedroom) leaves 19,571 rows.
Ridge vs RandomForest vs HistGradientBoosting, compared with 5-fold CV on log(price).

| Model | CV R² | Test R² | Test MAE (lakhs) |
|---|---|---|---|
| Ridge | 0.784 | 0.888 | 19.1 |
| **Random Forest** | **0.968** | **0.965** | **12.2** |
| Gradient Boosting | 0.963 | 0.961 | 13.1 |
