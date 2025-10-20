import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, messagebox
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import numpy as np
from src.neural_network import forward_propagation

# --- Global Variables for State Management ---
# Model parameters
W1, b1, W2, b2 = None, None, None, None
model_loaded = False

# Drawing state
last_x, last_y = None, None

# To prevent garbage collection of the image reference
input_photo = None

# --- Functions for GUI and Logic ---

def paint(event):
    """Draws on the canvas when the mouse is moved with the left button down."""
    global last_x, last_y
    x, y = event.x, event.y
    if last_x and last_y:
        # Draw on the visible canvas
        canvas.create_line((last_x, last_y, x, y),
                                width=20, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
        # Draw on the hidden PIL image
        draw.line([last_x, last_y, x, y], fill='white', width=20)
    last_x, last_y = x, y

def end_paint(event):
    """Resets the last mouse position when the button is released."""
    global last_x, last_y
    last_x, last_y = None, None

def clear_canvas():
    """Clears the canvas and resets the prediction labels."""
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill="black")
    result_label.config(text="Prediction: -")
    input_canvas.delete("all")
    for label in confidence_labels:
        label.config(text="0.0%")

def load_trained_model():
    """Loads the trained weights and biases from the .npz file."""
    global W1, b1, W2, b2, model_loaded
    try:
        data = np.load("../results/weights.npz")
        W1, b1 = data['W1'], data['b1']
        W2, b2 = data['W2'], data['b2']
        model_loaded = True
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        messagebox.showerror("Error", "Could not find 'weights.npz'.\nPlease run the training script and place the file here.")
        model_loaded = False

def visualize_input(img_array):
    """Displays the 28x28 preprocessed image in the GUI."""
    global input_photo
    img = Image.fromarray(img_array)
    img = img.resize((56, 56), Image.Resampling.NEAREST)
    input_photo = ImageTk.PhotoImage(image=img)
    input_canvas.create_image(0, 0, image=input_photo, anchor=tk.NW)

def predict_digit():
    """Performs prediction on the drawn digit."""
    if not model_loaded:
        messagebox.showwarning("Warning", "Model is not loaded. Cannot predict.")
        return

    # 1. Preprocess the image
    blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=1))
    img_resized = blurred_image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # 2. Visualize and Normalize
    visualize_input(img_array)
    img_normalized = img_array.astype("float32") / 255.0
    img_reshaped = img_normalized.reshape(784, 1) # Reshape for the network

    # 3. Perform forward propagation
    _, _, _, A2 = forward_propagation(img_reshaped, W1, b1, W2, b2)
    prediction = np.argmax(A2)

    # 4. Update the UI with results
    result_label.config(text=f"Prediction: {prediction}")
    update_confidence_labels(A2)

def update_confidence_labels(probabilities):
    """Updates the confidence percentage labels for each digit."""
    for i, prob in enumerate(probabilities.flatten()):
        percent = prob * 100
        confidence_labels[i].config(text=f"{percent:.1f}%")

# --- GUI Setup ---
root = tk.Tk()
root.title("Simple Digit Recognizer")
root.resizable(False, False)

# Main layout frames
main_frame = Frame(root, padx=10, pady=10)
main_frame.pack()
left_frame = Frame(main_frame)
left_frame.pack(side=tk.LEFT, padx=10)
right_frame = Frame(main_frame)
right_frame.pack(side=tk.RIGHT, padx=10)

# Left Frame: Drawing Canvas
Label(left_frame, text="Draw Here", font=("Helvetica", 12)).pack()
canvas = Canvas(left_frame, width=280, height=280, bg="black", cursor="cross")
canvas.pack(pady=5)

# Right Frame: Controls and Visualizations
control_frame = Frame(right_frame)
control_frame.pack(pady=10)
predict_button = Button(control_frame, text="Predict", command=predict_digit, width=10)
predict_button.pack(side=tk.LEFT, padx=5)
clear_button = Button(control_frame, text="Clear", command=clear_canvas, width=10)
clear_button.pack(side=tk.LEFT, padx=5)

# Prediction Result Label
result_label = Label(right_frame, text="Prediction: -", font=("Helvetica", 20, "bold"))
result_label.pack(pady=10)

# Confidence Percentage Labels
confidence_frame = Frame(right_frame, relief=tk.RIDGE, borderwidth=1)
confidence_frame.pack(pady=10, padx=5)
Label(confidence_frame, text="Confidence", font=("Helvetica", 12)).pack(pady=5)
confidence_labels = []
for i in range(10):
    bar_frame = Frame(confidence_frame)
    bar_frame.pack(fill=tk.X, padx=10, pady=2)
    Label(bar_frame, text=f"{i}:", font=("Helvetica", 10), width=2).pack(side=tk.LEFT)
    percent_label = Label(bar_frame, text="0.0%", font=("Helvetica", 10, "bold"), width=6, anchor='e')
    percent_label.pack(side=tk.RIGHT, padx=5)
    confidence_labels.append(percent_label)

# Network Input Visualization Canvas
Label(right_frame, text="Network Input (28x28)", font=("Helvetica", 12)).pack(pady=(15, 5))
input_canvas = Canvas(right_frame, width=56, height=56, bg="black")
input_canvas.pack()

# --- Backend Image and Event Bindings ---
# Create a hidden PIL image to draw on in the background
pil_image = Image.new("L", (280, 280), "black")
draw = ImageDraw.Draw(pil_image)

# Bind mouse events to functions
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", end_paint)

# --- Start Application ---
load_trained_model() # Load the model on startup
root.mainloop()      # Start the Tkinter event loop
