import pandas as pd
import numpy as np
import joblib

def main():
    try:
        model = joblib.load("./models/mood_best_model.joblib")
        print("Model loaded successfully")
        print("Pipeline steps:", model.named_steps)
    except Exception as e:
        print("Error loading model:", e)
        return
        
    try:
        tracks = pd.read_csv("./datasets/spotify_tracks_dataset.csv")
        print("Tracks loaded, shape:", tracks.shape)
        
        # Avoid missing columns that the model expects
        tracks = tracks.dropna(subset=['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo', 'duration_ms'])
        sub_focus = tracks[tracks['artists'].str.contains("Sub Focus", case=False, na=False) & 
                           tracks['track_name'].str.contains("Siren", case=False, na=False)]
        
        if sub_focus.empty:
            print("Could not find Sub Focus - Siren in data_lake/gold/tracks_prepared")
            return
            
        print("Found Sub Focus - Siren!")
        sub_focus = sub_focus.copy()
        sub_focus["spotify_url"] = "https://open.spotify.com/track/1234567890123456789012"
        
        from src.realtime_predictions import RealtimeMoodClassifier
        classifier = RealtimeMoodClassifier()
        
        classified = classifier.classify(sub_focus)
        
        print("\nPredictions for Sub Focus - Siren:")
        for _, row in classified.iterrows():
            print(f"Track: {row['track_name']} by {row['artists']}")
            print(f"Energy: {row['energy']:.3f}, Valence: {row['valence']:.3f}, Loudness: {row['loudness']:.3f}, Tempo: {row['tempo']:.3f}")
            print(f"Predicted Mood: {row['predicted_mood']}")
            for mood in ["sad", "happy", "energetic", "calm"]:
                print(f"  Proba {mood}: {row[f'proba_{mood}']:.3f}")
            print("-" * 40)
            
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
