import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from PIL import Image

# --- CONFIGURATION ---
# *** Ensure this path matches the folder containing your 75,000 images ***
CONSOLIDATED_FOLDER = 'existing_downloads' 
# *** Ensure this path matches your primary metadata file ***
CSV_PATH = 'master_dataset_clean11.csv' 
OUTPUT_EMBEDDING_FILE = 'image_embeddings_resnet50.npy'
OUTPUT_ID_FILE = 'image_ids_order.npy'

# ResNet50 input size
TARGET_SIZE = (224, 224) 
BATCH_SIZE = 64 # Adjust based on your GPU/RAM limits

def get_embedding_generator():
    """Loads ResNet50 pre-trained and strips the classification head."""
    print("Loading ResNet50 model for feature extraction...")
    # include_top=False strips the final classification layer
    # pooling='avg' collapses the final feature maps to a 2048-D vector
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

def load_and_preprocess_image(img_path):
    """Loads, converts to RGB, resizes, and prepares image for ResNet50."""
    try:
        # Use PIL to load and ensure RGB format
        img = Image.open(img_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        
        # Convert to numpy array and add batch dimension
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # ResNet50-specific preprocessing
        return preprocess_input(img_array)
    except Exception as e:
        # Failsafe for corrupted or unreadable images
        print(f"Error processing image {img_path}: {e}")
        return None


def generate_embeddings():
    # 1. Initialize the Embedding Generator Model
    model = get_embedding_generator()

    # 2. Get list of image files to process
    df = pd.read_csv(CSV_PATH)
    df['sample_id'] = df['sample_id'].astype(str) 
    
    # Expected filenames are in the format 'ID.jpg'
    all_files_expected = {f"{id_val}.jpg" for id_val in df['sample_id'].unique()}
    
    # Check what files actually exist on disk
    available_files = [
        f for f in os.listdir(CONSOLIDATED_FOLDER)
        if f.endswith('.jpg') and f in all_files_expected
    ]
    
    # Extract IDs (strip the .jpg extension)
    image_ids = [os.path.splitext(f)[0] for f in available_files]
    
    print(f"Found {len(available_files)} images matching the CSV for embedding.")
    
    # 3. Generate Embeddings in Batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(available_files), BATCH_SIZE), desc="Generating Embeddings"):
        batch_files = available_files[i:i + BATCH_SIZE]
        batch_inputs = []
        
        for filename in batch_files:
            img_path = os.path.join(CONSOLIDATED_FOLDER, filename)
            processed_img = load_and_preprocess_image(img_path)
            
            if processed_img is not None:
                batch_inputs.append(processed_img)

        if batch_inputs:
            batch_inputs_stacked = np.vstack(batch_inputs)
            # Predict features (embeddings)
            batch_embeddings = model.predict(batch_inputs_stacked, verbose=0)
            all_embeddings.append(batch_embeddings)

    # 4. Finalize and Save
    if all_embeddings:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Save the embedding matrix (N, 2048)
        np.save(OUTPUT_EMBEDDING_FILE, final_embeddings)
        
        # Save the corresponding IDs (to maintain order for mapping)
        np.save(OUTPUT_ID_FILE, np.array(image_ids))

        print("\n--- Embedding Generation Complete ---")
        print(f"Total Embeddings Generated: {final_embeddings.shape[0]}")
        print(f"Embedding Shape: {final_embeddings.shape}")
        print(f"Embeddings saved to: '{OUTPUT_EMBEDDING_FILE}'")
        print(f"IDs saved to: '{OUTPUT_ID_FILE}' (Crucial for merging with price data)")
    else:
        print("No embeddings could be generated. Check folder paths and image files.")

if __name__ == "__main__":
    generate_embeddings()