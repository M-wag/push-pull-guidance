import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import numpy as np

class ImageLabelingApp:
    def __init__(self, image_dir, questions, output_file="responses.json", max_display_size=(1200, 800)):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.questions = questions
        self.output_file = output_file
        self.max_display_size = max_display_size
        self.responses = {}
        self.current_index = 0

        self.root = tk.Tk()
        self.root.title("Image Labeling Tool")
        self.root.geometry(f"{max_display_size[0] + 200}x{max_display_size[1] + 400}")
        self.root.option_add('*Font', 'Arial 14')

        # Configure styles for larger UI elements
        style_config = {
            'font': ('Arial', 16),
            'padx': 15,
            'pady': 10
        }
        radio_style = {
            'font': ('Arial', 14),
            'padx': 10,
            'pady': 5
        }

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

        self.answer_vars = {}
        self.question_frames = []

        for question in questions:
            frame = tk.Frame(self.root)
            frame.pack(anchor="w", padx=20, pady=15)
            self.question_frames.append(frame)

            q_label = tk.Label(frame, text=question, **style_config)
            q_label.pack(side="left")

            var = tk.StringVar(value="No")
            self.answer_vars[question] = var

            yes_button = tk.Radiobutton(frame, text="Yes", variable=var, 
                                        value="Yes", **radio_style)
            no_button = tk.Radiobutton(frame, text="No", variable=var, 
                                       value="No", **radio_style)
            yes_button.pack(side="left", padx=10)
            no_button.pack(side="left", padx=10)

        self.submit_button = tk.Button(self.root, text="Submit", 
                                      command=self.submit, **style_config)
        self.submit_button.pack(pady=20)

        self.load_image()
        self.root.mainloop()

    def load_image(self):
        if self.current_index >= len(self.image_files):
            self.save_responses()
            messagebox.showinfo("Done", "All images have been labeled.")
            self.root.destroy()
            return

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        image = Image.open(image_path)

        # Optional duplication (remove if unnecessary)
        arr = np.concatenate((np.asarray(image), np.asarray(image)), axis=1)
        image = Image.fromarray(arr)

        # Calculate scaling factor to fit display area while maintaining aspect ratio
        width, height = image.size
        max_width, max_height = self.max_display_size
        
        # Calculate scaling factors for both dimensions
        width_ratio = max_width / width
        height_ratio = max_height / height
        
        # Use the smaller scaling factor to fit entire image
        scale_factor = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize the image using high-quality resampling
        image = image.resize((new_width, new_height), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo  # prevent garbage collection

        # Reset answers
        for var in self.answer_vars.values():
            var.set("No")

    def submit(self):
        image_name = self.image_files[self.current_index]
        self.responses[image_name] = {q: var.get() for q, var in self.answer_vars.items()}

        self.current_index += 1
        self.load_image()

    def save_responses(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.responses, f, indent=2)
        print(f"Responses saved to {self.output_file}")

if __name__ == "__main__":
    questions = [
        "Corrupted",
        "Changed",
        "Similar to template"
    ]
    image_dir = "data/data"
    ImageLabelingApp(image_dir, questions)
