# Porter Delivery Time Predictor

This is a simple Flask interface for the saved Porter neural-network model.

## Before running or deploying

Copy these files from the training notebook folder into this folder:

- `delivery_time_model.keras`
- `delivery_time_scaler.pkl`

Also save the exact feature order from the notebook and copy the result here:

```python
import joblib
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
```

`feature_columns.pkl` must contain 99 columns. The app can recover this list from
some fitted scikit-learn scalers, but saving it explicitly is the reliable option.

## Test locally

```powershell
conda activate dl_env
cd path\to\porter_delivery_app
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` and submit the form.
Depolyed link : https://eat-it-29.vercel.app/

## Deploy to Vercel

1. Put the three model artifacts in this folder. Do not upload the training dataset or notebook.
2. Push this folder to a GitHub repository.
3. In Vercel, choose **Add New → Project**, import the repository, and click **Deploy**.
4. Test the URL after the deployment finishes.

The entry point is the `app` object in `app.py`; current Vercel Flask support detects it automatically.
