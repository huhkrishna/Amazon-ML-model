import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights  # Import weights
from sklearn.metrics import f1_score  # Import f1_score
import src.constants as constants

class EntityExtractor:
    def __init__(self):
        # Load the pre-trained ResNet model
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Using ResNet with default weights
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

    def predict(self, image_path, entity_name):
        try:
            # Open and preprocess the image
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image file {image_path}: {e}")
            return ''

        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(image)  # Get model predictions

        return self.process_prediction(output, entity_name)

    def process_prediction(self, output, entity_name):
        # Process the model output
        predicted_value = output.squeeze().numpy()  # Convert to numpy array
        predicted_class = predicted_value.argmax()  # Get the index of the highest value

        allowed_units = constants.entity_unit_map.get(entity_name, None)
        if allowed_units is None:
            print(f"Invalid entity name: {entity_name}")
            return ''

        unit = list(allowed_units)[0]  # Defaulting to the first unit in the list
        prediction = f"{predicted_value[predicted_class]:.2f} {unit}"  # Format the number to 2 decimal places
        return prediction

    def generate_predictions(self, input_csv, output_csv, ground_truth_csv):
        # Read the input CSV and ground truth CSV
        df = pd.read_csv(input_csv)
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
        ground_truth_df = pd.read_csv(ground_truth_csv)
        ground_truth_df.columns = ground_truth_df.columns.str.strip()  # Strip spaces from ground truth columns

        # Print the ground truth DataFrame columns for debugging
        print("Ground Truth DataFrame Columns:", ground_truth_df.columns.tolist())

        predictions = []
        ground_truths = []

        for _, row in df.iterrows():
            image_filename = row['image_link'].split('/')[-1]
            image_path = os.path.join('images', image_filename)

            if os.path.exists(image_path):
                prediction = self.predict(image_path, row['entity_name'])
            else:
                print(f"Image not found: {image_path}")
                prediction = ''

            predictions.append(prediction)

            # Get the ground truth value from the ground_truth_df using image_link
            ground_truth = ground_truth_df.loc[ground_truth_df['image_link'] == row['image_link'], 'entity_value'].values
            if len(ground_truth) > 0:
                ground_truths.append(ground_truth[0])  # Append the first matched ground truth value
            else:
                ground_truths.append('')  # Append an empty string if no ground truth is found

        # Add predictions to the DataFrame
        df['prediction'] = predictions

        # Save the output CSV with the required structure
        if 'index' in df.columns:
            df[['index', 'prediction']].to_csv(output_csv, index=False)
        else:
            print("The 'index' column is missing from the DataFrame.")

        # Calculate F1 score
        f1 = f1_score(ground_truths, predictions, zero_division=0.0, average='weighted')
        print(f"F1 Score: {f1:.4f}")
