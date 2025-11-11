import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import gradio as gr

# === Load model ===
MODEL_PATH = "afb_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# === Constants ===
IMG_SIZE = (224, 224)
THRESHOLD = 0.45
POSITIVE_INDEX = 1
NEGATIVE_INDEX = 0

# === Preprocessing ===
eff_preprocess = keras.applications.efficientnet.preprocess_input

def preprocess(image: Image.Image):
    """Prepare image for model prediction."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    arr = eff_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image: Image.Image):
    """Run model prediction and return structured JSON."""
    try:
        x = preprocess(image)
        preds = model.predict(x)[0]

        prob_negative = float(preds[NEGATIVE_INDEX])
        prob_positive = float(preds[POSITIVE_INDEX])

        label = "AFB Positive" if prob_positive >= THRESHOLD else "AFB Negative"
        confidence = prob_positive if prob_positive >= THRESHOLD else prob_negative

        # Return clean JSON for frontend
        return {
            "label": label,
            "confidence": confidence,
            "prob_positive": prob_positive,
            "prob_negative": prob_negative,
            "threshold": THRESHOLD
        }

    except Exception as e:
        return {"error": str(e)}

# === Gradio Interface (synchronous mode) ===
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload AFB Smear Image"),
    outputs=gr.JSON(label="Prediction Result"),
    title="🧫 AFB Smear Detection AI",
    description="Upload a microscopic AFB smear image for AI-based TB detection.",
)

# === Launch (no queue, returns JSON immediately) ===
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
    )
