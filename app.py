from flask import Flask, render_template, request, session, g, redirect, url_for
import numpy as np
import sqlite3
import re
import librosa
import os
import csv
import uuid
import cv2
import time
import json
import hashlib
from typing import Optional
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import tensorflow as tf
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'KjhLJF54f6ds234H')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID', ''),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET', ''),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

DATABASE = "mydb.sqlite3"
MONGO_URI = os.getenv("MONGO_URI", "").strip()
PREDICTION_CACHE_TTL_SECONDS = 3600
prediction_cache = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== VIDEO/AUDIO MODELS ====================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TRAINING_CSV_PATHS = ["dataset.csv", "dataset_old.csv"]
MODEL_PATHS = {
    "audio": os.path.join(BASE_DIR, "model.keras"),
    "resnet": os.path.join(BASE_DIR, "resnet_model.h5"),
    "vgg": os.path.join(BASE_DIR, "vgg_model.h5"),
}

audio_model = None
video_model1 = None
video_model2 = None
model_load_error = None


def load_models_once():
    global audio_model, video_model1, video_model2, model_load_error

    if audio_model is not None and video_model1 is not None and video_model2 is not None:
        return True

    if model_load_error:
        return False

    missing = [path for path in MODEL_PATHS.values() if not os.path.exists(path)]
    if missing:
        model_load_error = (
            "Model files are missing on server: " + ", ".join(os.path.basename(path) for path in missing)
        )
        return False

    try:
        audio_model = tf.keras.models.load_model(MODEL_PATHS["audio"])
        video_model1 = load_model(MODEL_PATHS["resnet"])
        video_model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        video_model2 = load_model(MODEL_PATHS["vgg"])
        video_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return True
    except Exception as exception:
        model_load_error = f"Failed to load model files: {exception}"
        return False


class PredictionStore:
    def __init__(self):
        self.mongo_enabled = bool(MONGO_URI and MongoClient is not None)
        self._mongo_client = None
        self._mongo_collection = None

        if self.mongo_enabled:
            try:
                self._mongo_client = MongoClient(MONGO_URI)
                mongo_db = self._mongo_client["deepfake_detection"]
                self._mongo_collection = mongo_db["predictions"]
                self._mongo_collection.create_index("file_hash", unique=True)
            except Exception:
                self.mongo_enabled = False
                self._mongo_client = None
                self._mongo_collection = None

    def get_by_hash(self, file_hash: str) -> Optional[dict]:
        if self.mongo_enabled:
            doc = self._mongo_collection.find_one({"file_hash": file_hash})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc

        cursor = get_db().cursor()
        cursor.execute(
            """
            SELECT prediction_id, file_hash, original_name, stored_path, file_ext, media_type,
                   audio_result, video_result, effective_label, features_json, feedback_count,
                   created_at, updated_at
            FROM PREDICTIONS
            WHERE file_hash = ?
            """,
            (file_hash,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "prediction_id": row[0],
            "file_hash": row[1],
            "original_name": row[2],
            "stored_path": row[3],
            "file_ext": row[4],
            "media_type": row[5],
            "audio_result": row[6],
            "video_result": row[7],
            "effective_label": row[8],
            "features": json.loads(row[9]) if row[9] else [],
            "feedback_count": row[10] or 0,
            "created_at": row[11],
            "updated_at": row[12],
        }

    def get_by_prediction_id(self, prediction_id: str) -> Optional[dict]:
        if self.mongo_enabled:
            return self._mongo_collection.find_one({"prediction_id": prediction_id})

        cursor = get_db().cursor()
        cursor.execute(
            """
            SELECT prediction_id, file_hash, media_type, audio_result, video_result,
                   effective_label, features_json, feedback_count
            FROM PREDICTIONS
            WHERE prediction_id = ?
            """,
            (prediction_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "prediction_id": row[0],
            "file_hash": row[1],
            "media_type": row[2],
            "audio_result": row[3],
            "video_result": row[4],
            "effective_label": row[5],
            "features": json.loads(row[6]) if row[6] else [],
            "feedback_count": row[7] or 0,
        }

    def save_new(self, record: dict):
        if self.mongo_enabled:
            self._mongo_collection.insert_one(record)
            return

        cursor = get_db().cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO PREDICTIONS
            (prediction_id, file_hash, original_name, stored_path, file_ext, media_type,
             audio_result, video_result, effective_label, features_json, feedback_count,
             created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record["prediction_id"],
                record["file_hash"],
                record["original_name"],
                record["stored_path"],
                record["file_ext"],
                record["media_type"],
                record.get("audio_result"),
                record.get("video_result"),
                record.get("effective_label"),
                json.dumps(record.get("features", [])),
                record.get("feedback_count", 0),
                record["created_at"],
                record["updated_at"],
            ),
        )
        get_db().commit()

    def apply_feedback(self, prediction_id: str, corrected_label: str):
        now_ts = int(time.time())
        if self.mongo_enabled:
            self._mongo_collection.update_one(
                {"prediction_id": prediction_id},
                {
                    "$set": {
                        "effective_label": corrected_label,
                        "updated_at": now_ts,
                    },
                    "$inc": {"feedback_count": 1},
                },
            )
            return

        cursor = get_db().cursor()
        cursor.execute(
            """
            UPDATE PREDICTIONS
            SET effective_label = ?,
                feedback_count = COALESCE(feedback_count, 0) + 1,
                updated_at = ?
            WHERE prediction_id = ?
            """,
            (corrected_label, now_ts, prediction_id),
        )
        get_db().commit()


prediction_store = PredictionStore()


# ==================== DATABASE ====================
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


def init_local_tables():
    if prediction_store.mongo_enabled:
        return
    cursor = get_db().cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS PREDICTIONS (
            prediction_id TEXT PRIMARY KEY,
            file_hash TEXT UNIQUE,
            original_name TEXT,
            stored_path TEXT,
            file_ext TEXT,
            media_type TEXT,
            audio_result TEXT,
            video_result TEXT,
            effective_label TEXT,
            features_json TEXT,
            feedback_count INTEGER DEFAULT 0,
            created_at INTEGER,
            updated_at INTEGER
        )
        """
    )
    get_db().commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.before_request
def ensure_tables():
    init_local_tables()


# ==================== ROUTES ====================
@app.route('/')
def home():
    return render_template('index.html', background_image="/")


@app.route('/login.html')
def login():
    if 'user' in session:
        return redirect(url_for('model'))
    return render_template('login.html', background_image="/")


@app.route('/oauth/callback')
def oauth_callback():
    token = google.authorize_access_token()
    user = token.get('userinfo')
    if user:
        session['user'] = user
        session['Loggedin'] = True
        session['email'] = user.get('email')
        session['name'] = user.get('name')
        return redirect(url_for('model'))
    return redirect(url_for('login'))


@app.route('/auth/google')
def auth_google():
    redirect_uri = os.getenv('OAUTH_REDIRECT_URI', '').strip()
    if not redirect_uri:
        forwarded_proto = request.headers.get('X-Forwarded-Proto', request.scheme)
        redirect_uri = url_for('oauth_callback', _external=True, _scheme=forwarded_proto)
    return google.authorize_redirect(redirect_uri)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home1'))


@app.route('/contact.html')
def contact():
    return render_template('contact.html', background_image="/")


@app.route('/about.html')
def about():
    return render_template('about.html', background_image="/")


@app.route('/index.html')
def home1():

    @app.route('/')
    def home():
        return render_template('index.html', background_image="/")
    return render_template('index.html', background_image="/")

@app.route('/thankyou.html', methods=['GET', 'POST'])
def thank_you():
    return render_template('thankyou.html', background_image="/")


@app.route('/register.html', methods=['GET', 'POST'])
def signup():
    return redirect(url_for('auth_google'))


# ==================== UTILS FOR VIDEO ====================
def extract_audio_from_video(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(audio_path, format="wav")


def normalize_audio_to_wav(audio_path, output_wav_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_wav_path, format="wav")


def get_csv_header(path):
    if not os.path.exists(path):
        return []
    with open(path, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        return next(reader, [])


def extract_training_aligned_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_features = np.resize(mfcc_mean, 100)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_features = np.mean(mel_spec_db, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    chroma_features = np.resize(chroma_mean, 50)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    feature_vector = np.concatenate(
        [mfcc_features, mel_features, chroma_features, [zcr, spec_centroid, spec_flatness]]
    )
    return feature_vector.astype(float).tolist()


def append_features_to_training_csv(features, label):
    for csv_path in TRAINING_CSV_PATHS:
        header = get_csv_header(csv_path)
        if not header:
            continue
        expected_feature_count = max(0, len(header) - 1)
        aligned = np.resize(np.array(features, dtype=float), expected_feature_count).tolist()
        with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(aligned + [label])


def hash_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()


def get_cached_prediction(file_hash):
    cached = prediction_cache.get(file_hash)
    if not cached:
        return None
    if time.time() > cached["expires_at"]:
        prediction_cache.pop(file_hash, None)
        return None
    return cached["payload"]


def set_cached_prediction(file_hash, payload):
    prediction_cache[file_hash] = {
        "payload": payload,
        "expires_at": time.time() + PREDICTION_CACHE_TTL_SECONDS,
    }

def extract_audio_features_for_model(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5).T

    if mfccs.shape[0] < 21:
        pad_width = 21 - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    elif mfccs.shape[0] > 21:
        mfccs = mfccs[:21, :]

    return np.expand_dims(mfccs, axis=0)

def predict_audio_with_model(audio_path):
    if not load_models_once():
        raise RuntimeError(model_load_error or "Model loading failed")
    features = extract_audio_features_for_model(audio_path)
    prediction = audio_model.predict(features)[0][0]
    return "real" if prediction < 0.5 else "fake"

def extract_video_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)
            frames.append(frame)
        frame_count += 1

    cap.release()
    return np.concatenate(frames, axis=0) if frames else None

def predict_video_with_models(video_path):
    if not load_models_once():
        raise RuntimeError(model_load_error or "Model loading failed")
    frames = extract_video_frames(video_path, num_frames=10)
    if frames is None:
        return "error"

    vgg_preds = video_model2.predict(frames, verbose=0)
    resnet_preds = video_model1.predict(frames, verbose=0)

    vgg_class = np.argmax(vgg_preds, axis=1)
    resnet_class = np.argmax(resnet_preds, axis=1)

    frame_votes = []
    for idx in range(min(len(vgg_class), len(resnet_class))):
        frame_votes.append(np.argmax(np.bincount([vgg_class[idx], resnet_class[idx]])))

    if not frame_votes:
        return "error"
    final_prediction = np.argmax(np.bincount(frame_votes))

    return "real" if final_prediction == 0 else "fake"


# ==================== MODEL ROUTE ====================
@app.route('/model.html', methods=['GET', 'POST'])
@app.route('/model', methods=['GET', 'POST'])
def model():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    background_image = "/"
    feedback_visible = False
    audio_result, video_result, file_label = None, None, None
    msg = None
    prediction_id = None
    prediction_source = "model"

    if request.method == 'POST':
        if not load_models_once():
            return render_template(
                'model.html',
                background_image=background_image,
                feedback_visible=feedback_visible,
                msg=model_load_error or "Model loading failed",
            )

        uploaded_file = request.files.get('media_file')

        if not uploaded_file:
            return render_template('model.html', 
                                   background_image=background_image, 
                                   msg="No file uploaded")

        file_name = uploaded_file.filename or "uploaded_file"
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in ['.wav', '.mp3', '.mp4']:
            return render_template(
                'model.html',
                background_image=background_image,
                msg="Supported files: .wav, .mp3, .mp4",
                feedback_visible=False,
            )

        raw_bytes = uploaded_file.read()
        file_hash = hash_bytes(raw_bytes)

        cached_record = get_cached_prediction(file_hash)
        if cached_record:
            prediction_source = "memory cache"
            file_label = f"File: {file_name}"
            audio_result = cached_record.get("audio_result")
            video_result = cached_record.get("video_result")
            prediction_id = cached_record.get("prediction_id")
            feedback_visible = True

            return render_template(
                'model.html',
                file_label=file_label,
                audio_result=audio_result,
                video_result=video_result,
                background_image=background_image,
                feedback_visible=feedback_visible,
                prediction_id=prediction_id,
                prediction_source=prediction_source,
                msg=msg,
            )

        stored_record = prediction_store.get_by_hash(file_hash)
        if stored_record:
            prediction_source = "saved result"
            file_label = f"File: {file_name}"
            audio_result = stored_record.get("audio_result")
            video_result = stored_record.get("video_result")
            prediction_id = stored_record.get("prediction_id")
            feedback_visible = True

            set_cached_prediction(file_hash, stored_record)

            return render_template(
                'model.html',
                file_label=file_label,
                audio_result=audio_result,
                video_result=video_result,
                background_image=background_image,
                feedback_visible=feedback_visible,
                prediction_id=prediction_id,
                prediction_source=prediction_source,
                msg=msg,
            )

        save_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}{file_ext}")
        with open(save_path, "wb") as file_writer:
            file_writer.write(raw_bytes)

        file_label = f"File: {file_name}"
        media_type = "audio" if file_ext in ['.wav', '.mp3'] else "video"

        base_name = os.path.splitext(save_path)[0]
        audio_path = f"{base_name}.wav"

        if file_ext == '.wav':
            audio_path = save_path
            audio_result = predict_audio_with_model(audio_path)
            feedback_visible = True
        elif file_ext == '.mp3':
            normalize_audio_to_wav(save_path, audio_path)
            audio_result = predict_audio_with_model(audio_path)
            feedback_visible = True
        elif file_ext == '.mp4':
            extract_audio_from_video(save_path, audio_path)
            audio_result = predict_audio_with_model(audio_path)
            video_result = predict_video_with_models(save_path)
            feedback_visible = True
        else:
            msg = "Unsupported file type"

        extracted_features = extract_training_aligned_features(audio_path)
        prediction_id = str(uuid.uuid4())
        effective_label = video_result if video_result else audio_result
        now_ts = int(time.time())

        record = {
            "prediction_id": prediction_id,
            "file_hash": file_hash,
            "original_name": file_name,
            "stored_path": save_path,
            "file_ext": file_ext,
            "media_type": media_type,
            "audio_result": audio_result,
            "video_result": video_result,
            "effective_label": effective_label,
            "features": extracted_features,
            "feedback_count": 0,
            "created_at": now_ts,
            "updated_at": now_ts,
        }

        prediction_store.save_new(record)
        set_cached_prediction(file_hash, record)

        return render_template('model.html',
                               file_label=file_label,
                               audio_result=audio_result,
                               video_result=video_result,
                               background_image=background_image,
                               feedback_visible=feedback_visible,
                               prediction_id=prediction_id,
                               prediction_source=prediction_source,
                               msg=msg)

    return render_template('model.html', 
                           background_image=background_image, 
                           feedback_visible=feedback_visible)


# ==================== FEEDBACK ====================
@app.route('/feedback.html', methods=['POST'])
def feedback():
    prediction_id = request.form.get('prediction_id', '').strip()
    user_feedback = request.form.get('feedback', '').strip().lower()
    if not prediction_id or user_feedback not in {"yes", "no"}:
        return render_template('feedback.html', background_image="/")

    record = prediction_store.get_by_prediction_id(prediction_id)
    if not record:
        return render_template('feedback.html', background_image="/")

    predicted_label = (record.get("effective_label") or "real").lower()
    corrected_label = predicted_label if user_feedback == "yes" else ("fake" if predicted_label == "real" else "real")

    features = record.get("features") or []
    append_features_to_training_csv(features, corrected_label)
    prediction_store.apply_feedback(prediction_id, corrected_label)

    updated = prediction_store.get_by_hash(record.get("file_hash"))
    if updated:
        set_cached_prediction(record.get("file_hash"), updated)

    return render_template('feedback.html', background_image="/")


if __name__ == "__main__":
    app.run(debug=True)
