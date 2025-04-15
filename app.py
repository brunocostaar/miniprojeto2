import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = "d:\Trilha\Miniprojeto2\miniprojeto2\models\audio_emotion_model.keras"  # Example
SCALER_PATH = "d:\Trilha\Miniprojeto2\miniprojeto2\models\scaler.joblib"                # Example

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(data).T, axis=0)
    features.extend(zcr)

    # Chroma STFT
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0)
    features.extend(chroma)

    # MFCCs
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
    features.extend(mfccs)

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features.extend(rms)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    features.extend(mel)

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
st.title("Classificador de Emoções em Áudio")
st.write("Este aplicativo classifica emoções em áudio utilizando um modelo de aprendizado de máquina.") 

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    # Reproduzir o áudio enviado
    st.audio(temp_audio_path, format="audio/wav")

    # Extrair features
    features = extract_features(temp_audio_path)

    # Normalizar os dados com o scaler treinado
    features = scaler.transform(features)

    # Ajustar formato para o modelo
    features = np.expand_dims(features, axis=0)

    # Fazer a predição
    predictions = model.predict(features)

    # Exibir o resultado
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_emotion = EMOTIONS[predicted_class[0][0]]

    # Exibir probabilidades (gráfico de barras)
    st.subheader("Emoção Detectada!!")

    colors = ['#FF6F61', '#6B5B8A', '#88B04B', '#F7CAC9', '#92A8D1']
    classes = EMOTIONS
    fig, ax = plt.subplots()
    ax.set_ylabel('Probabilidade')
    ax.bar(classes, predictions[0], color=colors)
    st.pyplot(fig)


    # Remover o arquivo temporário
    os.remove(temp_audio_path)
