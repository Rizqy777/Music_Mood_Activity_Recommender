from __future__ import annotations

import base64
import html
from pathlib import Path

import gradio as gr

from src.recommender import MusicActivityRecommender

RECOMMENDER = MusicActivityRecommender()
ROOT = Path(__file__).resolve().parent


def load_banner_data() -> str:
  banner_path = ROOT / "images" / "ChatGPT Image 30 abr 2026, 13_11_02.png"
  if not banner_path.exists():
    return ""
  encoded = base64.b64encode(banner_path.read_bytes()).decode("ascii")
  return f"data:image/png;base64,{encoded}"


BANNER_DATA = load_banner_data()


def recommend(user_mood: str, activity: str, top_k: int):
    recommendations = RECOMMENDER.recommend(user_mood, activity, top_k=int(top_k))
    display_df = recommendations.rename(
        columns={
            "track_name": "Cancion",
            "artists": "Artista",
            "track_genre": "Genero",
            "target_mood": "Mood objetivo",
            "predicted_mood": "Mood detectado",
            "mood_confidence": "Confianza mood",
            "popularity": "Popularidad",
            "activity_interpreted_as": "Actividad interpretada",
            "recommendation_score": "Score",
            "reason": "Motivo",
            "spotify_url": "Spotify",
            "track_id": "Track ID",
        }
    )
    return render_cards(display_df), display_df


def render_cards(recommendations):
    cards = []
    for index, row in recommendations.iterrows():
        score = float(row.get("Score", 0.0))
        confidence = float(row.get("Confianza mood", 0.0))
        spotify_url = str(row.get("Spotify", ""))
        spotify_link = ""
        if spotify_url.startswith("https://open.spotify.com/"):
            spotify_link = (
                f'<a class="spotify-link" href="{html.escape(spotify_url)}" '
                f'target="_blank" rel="noopener noreferrer">Abrir en Spotify</a>'
            )
        cards.append(
            f"""
            <article class="song-card">
              <div class="rank">#{index + 1}</div>
              <div class="song-main">
                <h3>{html.escape(str(row.get("Cancion", "Cancion desconocida")))}</h3>
                <p class="artist">{html.escape(str(row.get("Artista", "Artista desconocido")))}</p>
                <div class="chips">
                  <span>{html.escape(str(row.get("Mood detectado", "-")))}</span>
                  <span>objetivo: {html.escape(str(row.get("Mood objetivo", "-")))}</span>
                  <span>{html.escape(str(row.get("Genero", "-")))}</span>
                  <span>{html.escape(str(row.get("Actividad interpretada", "-")))}</span>
                </div>
                <p class="reason">{html.escape(str(row.get("Motivo", "")))}</p>
                {spotify_link}
              </div>
              <div class="score-box">
                <strong>{score:.2f}</strong>
                <span>score</span>
                <small>{confidence:.2f} mood</small>
              </div>
            </article>
            """
        )
    return f'<section class="cards-grid">{"".join(cards)}</section>'


CSS = """
@import url("https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Space+Grotesk:wght@400;600;700&display=swap");

:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface-strong: #1f242c;
  --border: #2b313c;
  --muted: #b6beca;
  --accent: #f2b361;
  --accent-strong: #f2c57c;
  --accent-text: #1c1407;
  --highlight: #1db954;
  --font-display: "Fraunces", "Times New Roman", serif;
  --font-body: "Space Grotesk", "Trebuchet MS", sans-serif;
}

.gradio-container {
  max-width: 1540px !important;
  margin: 0 auto !important;
  font-family: var(--font-body) !important;
}
body, .gradio-container {
  background: radial-gradient(circle at 8% 20%, rgba(242, 179, 97, 0.15), transparent 40%),
    linear-gradient(180deg, #0b0f14 0%, #101521 100%) !important;
}
.hero {
  display: grid;
  grid-template-columns: minmax(240px, 1.1fr) minmax(280px, 1fr);
  gap: 28px;
  min-height: 240px;
  padding: 36px 38px;
  border: 1px solid var(--border);
  background:
    linear-gradient(135deg, rgba(36, 58, 94, 0.9), rgba(12, 14, 22, 0.98)),
    radial-gradient(circle at 82% 20%, rgba(242, 179, 97, 0.22), transparent 45%);
  border-radius: 14px;
  color: #f7f3ea;
  margin-bottom: 22px;
}
.hero h1 {
  font-family: var(--font-display);
  font-size: 44px;
  line-height: 1.1;
  margin: 0 0 12px 0;
  letter-spacing: 0.2px;
}
.hero p {
  font-size: 18px;
  max-width: 680px;
  color: #d9deea;
  margin: 0;
}
.hero-banner {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
}
.panel {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
  background: var(--surface);
}
.primary-button button {
  min-height: 52px !important;
  font-size: 17px !important;
  font-weight: 700 !important;
  background: var(--accent) !important;
  color: var(--accent-text) !important;
  border: 0 !important;
}
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 14px;
  margin-top: 8px;
}
.song-card {
  position: relative;
  min-height: 172px;
  border: 1px solid var(--border);
  background: var(--surface-strong);
  border-radius: 12px;
  padding: 18px 18px 18px 68px;
  color: #f5f0e7;
  display: flex;
  justify-content: space-between;
  gap: 18px;
}
.rank {
  position: absolute;
  left: 18px;
  top: 18px;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: var(--accent-strong);
  color: #171717;
  display: grid;
  place-items: center;
  font-weight: 800;
}
.song-main h3 {
  font-size: 22px;
  margin: 0 0 6px;
  line-height: 1.15;
}
.artist {
  color: #c7ccd6;
  margin: 0 0 12px;
  font-size: 15px;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}
.chips span {
  border: 1px solid #3a4250;
  background: #222630;
  color: #e9edf5;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 13px;
}
.reason {
  color: #b9c0cc;
  margin: 0;
  font-size: 14px;
  line-height: 1.35;
}
.spotify-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 12px;
  min-height: 34px;
  padding: 0 13px;
  border-radius: 999px;
  background: var(--highlight);
  color: #07130b !important;
  font-weight: 800;
  text-decoration: none !important;
  width: fit-content;
}
.score-box {
  min-width: 86px;
  align-self: stretch;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-left: 1px solid var(--border);
  padding-left: 14px;
}
.score-box strong {
  font-size: 30px;
  line-height: 1;
  color: var(--accent-strong);
}
.score-box span,
.score-box small {
  color: #b9c0cc;
}
@media (max-width: 980px) {
  .hero {
    grid-template-columns: 1fr;
  }
}
"""


with gr.Blocks(title="Music Mood Activity Recommender") as demo:
    gr.HTML(
        f"""
        <section class="hero">
          <div>
            <h1>Music Mood Activity Recommender</h1>
            <p>Recomendaciones que combinan como te sientes, que vas a hacer y las caracteristicas acusticas de cada cancion.</p>
          </div>
          <div>
            <img class="hero-banner" src="{BANNER_DATA}" alt="Banner del recomendador" />
          </div>
        </section>
        """
    )
    with gr.Group(elem_classes=["panel"]):
        with gr.Row():
            mood = gr.Dropdown(
                choices=["triste", "feliz", "energico", "tranquilo"],
                value="triste",
                label="Como te sientes?",
                scale=1,
            )
            activity = gr.Textbox(
                value="quiero llorar",
                label="Que actividad o momento tienes ahora?",
                placeholder="Ejemplo: quiero llorar, voy al gym, estudiar, limpiar la casa...",
                scale=2,
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Numero de canciones", scale=1)
        button = gr.Button("Buscar canciones", elem_classes=["primary-button"])

    cards = gr.HTML(label="Recomendaciones")
    output = gr.Dataframe(label="Detalle tecnico", wrap=True)
    button.click(recommend, inputs=[mood, activity, top_k], outputs=[cards, output])

    gr.Examples(
        examples=[
            ["triste", "quiero llorar", 5],
            ["triste", "quiero fregar el suelo", 5],
            ["triste", "voy a entrenar en el gym", 5],
            ["feliz", "voy a estudiar programacion", 5],
            ["tranquilo", "quiero dormir", 5],
            ["energico", "voy a limpiar la casa", 5],
        ],
        inputs=[mood, activity, top_k],
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base())
