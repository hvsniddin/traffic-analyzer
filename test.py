from ultralytics import YOLO

# 1. Load your existing model
model = YOLO("weights/test.pt")

# 2. Check the current names to ensure you map the IDs correctly
print("Old class names:", model.names)

# 3. Define your new class names 
# Make sure the keys (0-7) match the exact order of the original classes
new_names = {
    0: "bicycle",
    1: "bus",
    2: "car",
    3: "greenlight",
    4: "motorcycle",
    5: "person",
    6: "redlight",
    7: "truck"
}

# 4. Update the internal PyTorch model; YOLO.names is read-only
model.model.names = new_names

# 5. Save the updated model to a new file
model.save("weights/test.pt")

print("Model saved successfully with new class names!")