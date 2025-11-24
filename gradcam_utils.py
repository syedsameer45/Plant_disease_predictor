# gradcam_utils.py
"""
Grad-CAM utilities: preprocessing, functional grad-model builder, heatmap, overlay and base64 export.
"""

from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf

def preprocess_pil(img_pil, target_size=(225, 225)):
    """Convert PIL image to (1,H,W,3) float32 in range [0,1]."""
    img = img_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def find_last_conv_layer(model):
    """Return the name of the last Conv2D layer in model, or None."""
    for layer in reversed(model.layers):
        try:
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        except Exception:
            pass
        lname = getattr(layer, "name", "").lower()
        if "conv" in lname:
            return layer.name
    return None

def build_functional_grad_model(model, last_conv_layer_name, input_shape=(225,225,3)):
    """
    Build a functional model that maps input -> [last_conv_output, predictions]
    by replaying layers on a fresh Input tensor (weights are shared).
    """
    from tensorflow.keras import Input, Model # type: ignore

    inp = Input(shape=input_shape)
    x = inp
    last_conv_output = None
    found = False

    for layer in model.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x
            found = True
            break

    if not found:
        raise ValueError(f"Last conv layer '{last_conv_layer_name}' not found in model.layers")

    y = last_conv_output
    after = False
    for layer in model.layers:
        if after:
            y = layer(y)
        if layer.name == last_conv_layer_name:
            after = True

    grad_model = Model(inputs=inp, outputs=[last_conv_output, y])
    return grad_model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, blur_radius=2):
    """
    Robust Grad-CAM:
      - img_array: (1,H,W,3) float32
      - model: original keras model
      - last_conv_layer_name: string
      - pred_index: class index (optional)
    Returns heatmap (H,W) float32 in 0..1.
    """
    # Build grad_model by replaying layers on a fresh Input
    grad_model = build_functional_grad_model(model, last_conv_layer_name,
                                            input_shape=(img_array.shape[1], img_array.shape[2], img_array.shape[3]))
    # Ensure built
    try:
        _ = grad_model(np.zeros(img_array.shape, dtype=np.float32))
    except Exception:
        pass

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients returned None — cannot compute Grad-CAM.")

    grads_np = np.array(grads)
    max_abs_grad = float(np.max(np.abs(grads_np)))
    if max_abs_grad < 1e-8:
        pooled_grads = np.mean(np.abs(grads_np), axis=(0,1,2))
    else:
        pooled_grads = np.array(tf.reduce_mean(grads, axis=(0,1,2)).numpy())

    conv_outputs_np = np.array(conv_outputs[0].numpy())  # (h,w,filters)

    weights = pooled_grads.reshape((1,1,-1))
    weighted_maps = conv_outputs_np * weights
    heatmap = np.sum(weighted_maps, axis=-1)

    # ReLU + normalize
    heatmap = np.maximum(heatmap, 0.0)
    minv, maxv = float(np.min(heatmap)), float(np.max(heatmap))
    if maxv - minv <= 1e-8:
        heatmap = np.mean(np.abs(conv_outputs_np), axis=-1)
        minv, maxv = float(np.min(heatmap)), float(np.max(heatmap))
        if maxv - minv <= 1e-8:
            return np.zeros_like(heatmap, dtype=np.float32)

    heatmap = (heatmap - minv) / (maxv - minv + 1e-12)

    heat_pil = Image.fromarray(np.uint8(heatmap * 255))
    heat_pil = heat_pil.resize((img_array.shape[2], img_array.shape[1]), resample=Image.BILINEAR)
    if blur_radius > 0:
        heat_pil = heat_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    heatmap = np.array(heat_pil).astype("float32") / 255.0
    return heatmap

def overlay_heatmap(original_pil, heatmap, alpha=0.4, cmap="jet"):
    """Overlay heatmap (0..1) onto original PIL image, return PIL image."""
    import matplotlib.cm as cm
    orig = np.array(original_pil.convert("RGB")).astype("float32")/255.0
    color_map = cm.get_cmap(cmap)
    colored = color_map(heatmap)[:, :, :3]
    if colored.shape[:2] != orig.shape[:2]:
        from PIL import Image as PILImage
        colored = np.array(PILImage.fromarray((colored*255).astype("uint8")).resize((orig.shape[1], orig.shape[0]))).astype("float32")/255.0
    overlay = orig*(1-alpha) + colored*alpha
    overlay = np.clip(overlay, 0, 1)
    return Image.fromarray((overlay*255).astype("uint8"))

def pil_to_base64(pil_img, fmt="JPEG"):
    """Convert PIL image to base64 string (no data URI prefix)."""
    import base64
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64
