import librosa
import numpy as np


def analizar_spotify_features(ruta_audio):

    print("\n--- ANALISIS TIPO SPOTIFY ---")

    # Cargar audio (30s para velocidad)
    y, sr = librosa.load(ruta_audio, duration=30)

    # =========================
    # 1. TEMPO (BPM)
    # =========================
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # FIX: librosa puede devolver array o float
    if isinstance(tempo, (np.ndarray, list)):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # =========================
    # 2. LOUDNESS (dB aprox)
    # =========================
    rms = librosa.feature.rms(y=y)[0]
    loudness = 20 * np.log10(np.mean(rms) + 1e-9)

    # =========================
    # 3. ENERGY
    # =========================
    energy = float(np.mean(rms))

    # =========================
    # 4. BRIGHTNESS (Spectral Centroid)
    # =========================
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))

    # =========================
    # 5. SPEECHINESS (proxy)
    # =========================
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    speechiness = float(np.mean(np.var(mfcc, axis=1)))

    # =========================
    # 6. ACOUSTICNESS (proxy)
    # =========================
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

    # =========================
    # 7. INSTRUMENTALNESS (proxy)
    # =========================
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)

    instrumentalness = float(
        np.mean(np.abs(harmonic)) /
        (np.mean(np.abs(percussive)) + 1e-9)
    )

    # =========================
    # 8. LIVENESS (proxy)
    # =========================
    liveness = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    # =========================
    # 9. VALENCE (proxy emocional)
    # =========================
    valence = float(
        (spectral_centroid / (sr / 2)) *
        (1 - spectral_flatness)
    )

    # =========================
    # 10. DANCEABILITY (proxy)
    # =========================
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    beat_consistency = len(beat_frames) / (len(onset_env) + 1e-9)

    danceability = float(beat_consistency * energy)

    # =========================
    # OUTPUT
    # =========================
    print(f"Tempo (BPM): {tempo:.2f}")
    print(f"Energy: {energy:.4f}")
    print(f"Loudness (dB aprox): {loudness:.2f}")
    print(f"Brightness: {spectral_centroid:.2f}")
    print(f"Speechiness: {speechiness:.4f}")
    print(f"Acousticness: {spectral_flatness:.4f}")
    print(f"Instrumentalness: {instrumentalness:.4f}")
    print(f"Liveness: {liveness:.4f}")
    print(f"Valence (proxy): {valence:.4f}")
    print(f"Danceability (proxy): {danceability:.4f}")


# =========================
# EJEMPLO DE USO
# =========================

analizar_spotify_features(
    r"scripts\Boom Breaks - Drug is Fuck (Original mix).wav"
)