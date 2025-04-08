import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add
from tensorflow.keras.optimizers import Adam

# Load the pre-trained ResNet50 model for feature extraction
def extract_features(image_path):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    features = model.predict(img_array)
    return features

# Prepare the dataset (dummy dataset for demonstration)
# Replace this with your actual dataset
images = ['path/to/image1.jpg', 'path/to/image2.jpg']  # Add your image paths
captions = ['A cat sitting on a chair.', 'A dog playing in the park.']  # Corresponding captions

# Tokenize the captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Prepare the sequences
sequences = tokenizer.texts_to_sequences(captions)
sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split into input (features) and output (captions)
X_img = np.array([extract_features(img) for img in images])
X_cap = sequences[:, :-1]  # All but last word
y_cap = sequences[:, 1:]    # All but first word

# Define the model
def create_model(vocab_size, max_length):
    # Image feature input
    image_input = Input(shape=(2048,))
    image_dense = Dense(256, activation='relu')(image_input)

    # Caption input
    caption_input = Input(shape=(max_length - 1,))
    caption_embedding = Embedding(vocab_size, 256)(caption_input)
    caption_lstm = LSTM(256)(caption_embedding)

    # Combine image and caption features
    decoder_input = Add()([image_dense, caption_lstm])
    output = Dense(vocab_size, activation='softmax')(decoder_input)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model

# Compile the model
model = create_model(vocab_size, max_length)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit([X_img, X_cap], np.expand_dims(y_cap, -1), epochs=100, verbose=1)

# Function to generate captions for new images
def generate_caption(image_path):
    features = extract_features(image_path)
    caption = [tokenizer.word_index['<start>']]  # Start token
    for _ in range(max_length):
        sequence = pad_sequences([caption], maxlen=max_length - 1, padding='post')
        prediction = model.predict([features, sequence])
        predicted_id = np.argmax(prediction)
        caption.append(predicted_id)
        if predicted_id == tokenizer.word_index['<end>']:  # End token
            break
    return tokenizer.sequences_to_texts([caption])[0]

# Example usage
new_image_path = 'path/to/new_image.jpg'  # Replace with your image path
caption = generate_caption(new_image_path)
print("Generated Caption:", caption)
