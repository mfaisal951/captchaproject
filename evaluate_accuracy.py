# evaluate_accuracy.py
# Script to evaluate model accuracy on all test images

import os
from captcha import Captcha

def evaluate_model_accuracy():
    """Evaluate the model accuracy on all test images"""
    
    input_dir = 'input'
    output_dir = 'output'
    
    # Initialize the CAPTCHA recognizer
    recognizer = Captcha()
    
    total_images = 0
    correct_predictions = 0
    character_correct = 0
    total_characters = 0
    
    results = []
    
    print("Evaluating model accuracy on all test images...")
    print("-" * 60)
    
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.jpg'):
            continue
            
        img_path = os.path.join(input_dir, fname)
        label_fname = fname.replace('input', 'output').replace('.jpg', '.txt')
        label_path = os.path.join(output_dir, label_fname)
        
        if not os.path.exists(label_path):
            continue
            
        # Read true label
        with open(label_path, 'r') as f:
            true_label = f.read().strip()
        
        # Predict
        temp_output = 'temp_pred.txt'
        recognizer(img_path, temp_output)
        
        with open(temp_output, 'r') as f:
            predicted_label = f.read().strip()
        
        # Clean up temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        # Calculate character-level accuracy
        char_correct = sum(1 for i in range(min(len(true_label), len(predicted_label))) 
                          if true_label[i] == predicted_label[i])
        character_correct += char_correct
        total_characters += len(true_label)
        
        # Check if full captcha is correct
        is_correct = (true_label == predicted_label)
        if is_correct:
            correct_predictions += 1
        
        total_images += 1
        
        # Store result
        status = "✓" if is_correct else "✗"
        results.append({
            'image': fname,
            'true': true_label,
            'predicted': predicted_label,
            'correct': is_correct,
            'char_accuracy': char_correct / len(true_label) * 100
        })
        
        print(f"{fname}: {true_label} → {predicted_label} {status}")
    
    # Calculate overall accuracies
    full_captcha_accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    character_accuracy = (character_correct / total_characters) * 100 if total_characters > 0 else 0
    
    print("-" * 60)
    print("ACCURACY SUMMARY:")
    print(f"Full Captcha Accuracy: {full_captcha_accuracy:.2f}% ({correct_predictions}/{total_images})")
    print(f"Character-Level Accuracy: {character_accuracy:.2f}% ({character_correct}/{total_characters})")
    
    # Show detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 60)
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"{result['image']:12} | True: {result['true']} | Pred: {result['predicted']} | {status} | Char Acc: {result['char_accuracy']:.1f}%")
    
    # Show incorrect predictions
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print(f"\nINCORRECT PREDICTIONS ({len(incorrect)}):")
        print("-" * 60)
        for result in incorrect:
            print(f"{result['image']}: {result['true']} → {result['predicted']}")
    
    return full_captcha_accuracy, character_accuracy

if __name__ == "__main__":
    evaluate_model_accuracy() 