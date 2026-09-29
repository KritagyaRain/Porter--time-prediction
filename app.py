from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "delivery_time_model.keras"
SCALER_PATH = BASE_DIR / "delivery_time_scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

app = Flask(__name__)


def load_feature_columns(scaler):
    """Use the saved column list, or recover it from a fitted sklearn scaler."""
    if FEATURE_COLUMNS_PATH.exists():
        return joblib.load(FEATURE_COLUMNS_PATH)
    if hasattr(scaler, "feature_names_in_"):
        return scaler.feature_names_in_.tolist()
    raise RuntimeError(
        "feature_columns.pkl is missing. Save X.columns.tolist() from the "
        "training notebook and store it as feature_columns.pkl."
    )


if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise RuntimeError(
        "Missing model files. Copy delivery_time_model.keras and "
        "delivery_time_scaler.pkl into the porter_delivery_app folder."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
feature_columns = load_feature_columns(scaler)


def category_values(column_name):
    """Get the categorical values used in training from one-hot column names."""
    prefix = f"{column_name}_"
    return sorted(
        column[len(prefix):] for column in feature_columns if column.startswith(prefix)
    )


def make_prediction(form):
    raw_input = pd.DataFrame([{
        "market_id": float(form["market_id"]),
        "store_primary_category": form["store_primary_category"],
        "order_protocol": float(form["order_protocol"]),
        "total_items": float(form["total_items"]),
        "subtotal": float(form["subtotal"]),
        "num_distinct_items": float(form["num_distinct_items"]),
        "min_item_price": float(form["min_item_price"]),
        "max_item_price": float(form["max_item_price"]),
        "total_onshift_partners": float(form["total_onshift_partners"]),
        "total_busy_partners": float(form["total_busy_partners"]),
        "total_outstanding_orders": float(form["total_outstanding_orders"]),
        "order_hour": int(form["order_hour"]),
        "order_day": int(form["order_day"]),
        "is_weekend": int(form["is_weekend"]),
        "order_month": int(form["order_month"]),
    }])

    encoded_input = pd.get_dummies(
        raw_input,
        columns=["market_id", "store_primary_category", "order_protocol"],
        dtype=int,
    )
    encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)
    scaled_input = scaler.transform(encoded_input)
    return round(float(model.predict(scaled_input, verbose=0)[0][0]), 2)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            prediction = make_prediction(request.form)
        except (KeyError, TypeError, ValueError) as exc:
            error = f"Please enter valid values in every field. ({exc})"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        markets=category_values("market_id"),
        categories=category_values("store_primary_category"),
        protocols=category_values("order_protocol"),
    )


if __name__ == "__main__":
    app.run(debug=True)
