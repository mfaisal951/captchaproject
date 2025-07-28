# predict_example.py
# Example script to demonstrate CAPTCHA prediction

import os
from captcha import Captcha

def main():
    # Initialize the CAPTCHA recognizer
    recognizer = Captcha()
    
    # Find the first image in the input directory
    input_dir = 'input'
    sample_images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    if not sample_images:
        print("No images found in input directory.")
        return
    
    # Predict CAPTCHA for the first image
    sample_image_path = os.path.join(input_dir, sample_images[0])
    output_path = 'predicted_output.txt'
    
    print(f"Predicting CAPTCHA for image: {sample_image_path}")
    
    try:
        # Use the Captcha class to predict
        recognizer(sample_image_path, output_path)
        print(f"Prediction completed! Result saved to {output_path}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
