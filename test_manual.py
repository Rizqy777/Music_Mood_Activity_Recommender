import pandas as pd
import joblib
from src.realtime_predictions import RealtimeMoodClassifier

def main():
    # Simulate the user's manual input
    df = pd.DataFrame([{
        "spotify_url": "https://open.spotify.com/track/1234567890123456789012",
        "track_name": "Test Drum and Bass",
        "artists": "Test Artist",
        "track_genre": "dnb",
        "popularity": 50,
        "explicit": False,
        "danceability": 0.9,
        "energy": 1.0,
        "speechiness": 0.05,
        "acousticness": 0.0,
        "instrumentalness": 0.5,
        "liveness": 0.1,
        "valence": 0.0,  # Low valence
        "loudness": -3.0,
        "tempo": 142.0,
        "duration_ms": 200000
    }])
    
    classifier = RealtimeMoodClassifier()
    classified = classifier.classify(df)
    
    for _, row in classified.iterrows():
        print(f"Predicted Mood: {row['predicted_mood']}")
        for mood in ["sad", "happy", "energetic", "calm"]:
            print(f"  Proba {mood}: {row[f'proba_{mood}']:.3f}")

if __name__ == "__main__":
    main()
