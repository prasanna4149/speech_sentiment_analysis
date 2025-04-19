import os
import tempfile
import pickle
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import model_from_json, load_model
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# ✅ Load CNN model architecture
try:
    with open("CNN_model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load CNN model weights
    loaded_model.load_weights("CNN_model.weights.h5")
    loaded_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("✅ CNN model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading CNN model: {e}")

# ✅ Load Fully Trained Model (if needed)
try:
    trained_model = load_model("trained_model.h5")
    print("✅ Fully trained model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading trained model: {e}")

# ✅ Load preprocessing objects
try:
    with open("encoder2.pickle", "rb") as f:
        encoder2 = pickle.load(f)
    print("✅ Label encoder loaded successfully!")
except Exception as e:
    print(f"❌ Error loading encoder2: {e}")

try:
    with open("scaler2.pickle", "rb") as f:
        scaler2 = pickle.load(f)
    print("✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading scaler2: {e}")

print("🚀 All models and preprocessing objects loaded successfully!")


# ✅ Feature Extraction Functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512, target_size=2376):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))

    # Adjust feature vector size
    if len(result) < target_size:
        # Pad with zeros if smaller
        result = np.pad(result, (0, target_size - len(result)), mode='constant')
    else:
        # Truncate if larger
        result = result[:target_size]

    return result


def get_predict_feat(path):
    """Loads an audio file, extracts features, scales, and reshapes it for prediction."""
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    result = np.reshape(result, (1, 2376))  # Ensures correct shape
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)

    return final_result


# ✅ Emotion Mapping
emotions1 = {1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad", 5: "Angry", 6: "Fear", 7: "Disgust", 8: "Surprise"}


def prediction(path1):
    """Runs the prediction on the given audio file."""
    res = get_predict_feat(path1)
    predictions = loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    print(f"🎤 Predicted Emotion: {y_pred[0][0]}")
    return y_pred[0][0]

@csrf_exempt
def predict_audio(request):
    """Handles file upload and returns predicted emotion"""
    if request.method == "POST" and request.FILES.get("audio"):
        uploaded_file = request.FILES["audio"]

        # Save file temporarily
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        try:
            with open(file_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            print(f"✅ Audio file saved: {file_path}")

            # Predict emotion
            predicted_emotion = prediction(file_path)

            # ✅ Send JSON Response to Frontend
            response_data = {
                "status": "success",
                "emotion": predicted_emotion
            }
            print(f"✅ Sending response: {response_data}")  # Debugging purpose

            return JsonResponse(response_data, status=200)

        except Exception as e:
            print(f"❌ Error: {e}")
            return JsonResponse({"status": "error", "message": "Internal Server Error"}, status=500)

        finally:
            # Delete temp file
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return JsonResponse({"status": "error", "message": "No file uploaded"}, status=400)
