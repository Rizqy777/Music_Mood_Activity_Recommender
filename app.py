from __future__ import annotations

import base64
import html
import random
import secrets
from pathlib import Path

import gradio as gr

from src.recommender import MusicActivityRecommender, normalize_mood

RECOMMENDER = MusicActivityRecommender()
ROOT = Path(__file__).resolve().parent
POOL_MULTIPLIER = 30
MIN_POOL_SIZE = 50
MAX_POOL_SIZE = 1000
SCORE_RANGES = {
    "0.00 - 1.00": (0.0, 1.01),
    "0.75 - 1.00": (0.75, 1.01),
    "0.60 - 0.75": (0.60, 0.75),
    "0.45 - 0.60": (0.45, 0.60),
}
GENRE_CHOICES = [
    "Todos",
    "acoustic",
    "afrobeat",
    "alt-rock",
    "alternative",
    "ambient",
    "anime",
    "black-metal",
    "bluegrass",
    "blues",
    "brazil",
    "breakbeat",
    "british",
    "cantopop",
    "chicago-house",
    "children",
    "chill",
    "classical",
    "club",
    "comedy",
    "country",
    "dance",
    "dancehall",
    "death-metal",
    "deep-house",
    "detroit-techno",
    "disco",
    "disney",
    "drum-and-bass",
    "dub",
    "dubstep",
    "edm",
    "electro",
    "electronic",
    "emo",
    "folk",
    "forro",
    "french",
    "funk",
    "garage",
    "german",
    "gospel",
    "goth",
    "grindcore",
    "groove",
    "grunge",
    "guitar",
    "happy",
    "hard-rock",
    "hardcore",
    "hardstyle",
    "heavy-metal",
    "hip-hop",
    "honky-tonk",
    "house",
    "idm",
    "indian",
    "indie-pop",
    "indie",
    "industrial",
    "iranian",
    "j-dance",
    "j-idol",
    "j-pop",
    "j-rock",
    "jazz",
    "k-pop",
    "kids",
    "latin",
    "latino",
    "malay",
    "mandopop",
    "metal",
    "metalcore",
    "minimal-techno",
    "mpb",
    "new-age",
    "opera",
    "pagode",
    "party",
    "piano",
    "pop-film",
    "pop",
    "power-pop",
    "progressive-house",
    "psych-rock",
    "punk-rock",
    "punk",
    "r-n-b",
    "reggae",
    "reggaeton",
    "rock-n-roll",
    "rock",
    "rockabilly",
    "romance",
    "sad",
    "salsa",
    "samba",
    "sertanejo",
    "show-tunes",
    "singer-songwriter",
    "ska",
    "sleep",
    "songwriter",
    "soul",
    "spanish",
    "study",
    "swedish",
    "synth-pop",
    "tango",
    "techno",
    "trance",
    "trip-hop",
    "turkish",
    "world-music",
    "spotify_unknown",
]


def load_banner_data() -> str:
    banner_path = ROOT / "images" / "ChatGPT Image 30 abr 2026, 13_11_02.png"
    if not banner_path.exists():
        return ""
    encoded = base64.b64encode(banner_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


BANNER_DATA = load_banner_data()


def sample_recommendations(pool, top_k: int, seed: int):
    if pool is None or pool.empty:
        return pool
    rng = random.Random(seed)
    indices = list(pool.index)
    rng.shuffle(indices)
    selected = pool.loc[indices[:top_k]].copy()
    return selected.reset_index(drop=True)


def filter_by_score_range(pool, score_range: str):
    if pool is None or len(pool) == 0:
        return pool
    if "recommendation_score" not in pool.columns:
        return pool
    min_score, max_score = SCORE_RANGES.get(score_range, SCORE_RANGES["0.00 - 1.00"])
    filtered = pool[
        (pool["recommendation_score"].astype(float) >= min_score)
        & (pool["recommendation_score"].astype(float) < max_score)
    ].copy()
    return filtered


def filter_by_mood_match(pool, user_mood: str):
    if pool is None or len(pool) == 0:
        return pool
    if "predicted_mood" not in pool.columns:
        return pool
    normalized = normalize_mood(user_mood)
    return pool[pool["predicted_mood"].astype(str).str.lower() == normalized].copy()




def format_recommendations(recommendations):
    recommendations = recommendations.copy()
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
            "audio_predicted_mood": "Mood audio",
            "lyrics_predicted_mood": "Mood letra",
            "mood_contrast": "Contraste audio/letra",
        }
    )
    warning = None
    if "Actividad interpretada" in display_df.columns:
        if (display_df["Actividad interpretada"] == "actividad_general").any():
            warning = "No se detecta actividad clara. Escribe algo mas especifico."
    return render_cards(display_df, warning=warning), display_df


def get_recommendation_ids(frame):
    if frame is None or len(frame) == 0:
        return []
    if "track_id" in frame.columns:
        return frame["track_id"].astype(str).tolist()
    return frame.index.astype(str).tolist()


def filter_unseen(pool, seen_ids):
    if pool is None or len(pool) == 0:
        return pool
    if not seen_ids:
        return pool
    seen_set = {str(item) for item in seen_ids}
    if "track_id" in pool.columns:
        mask = ~pool["track_id"].astype(str).isin(seen_set)
    else:
        mask = ~pool.index.astype(str).isin(seen_set)
    return pool[mask].copy()


def build_pool(user_mood: str, activity: str, pool_size: int, artist_filter: str, genre_filter: str):
    return RECOMMENDER.recommend(
        user_mood,
        activity,
        top_k=pool_size,
        artist_filter=artist_filter,
        genre_filter=genre_filter,
    )


def recommend(
    user_mood: str,
    activity: str,
    artist_filter: str,
    genre_filter: str,
    top_k: int,
    score_range: str,
):
    recommendations = RECOMMENDER.recommend(
        user_mood,
        activity,
        top_k=int(top_k),
        artist_filter=artist_filter,
        genre_filter=genre_filter,
    )
    recommendations = filter_by_mood_match(recommendations, user_mood)
    recommendations = filter_by_score_range(recommendations, score_range)
    return format_recommendations(recommendations)


def recommend_with_pool(
    user_mood: str,
    activity: str,
    artist_filter: str,
    genre_filter: str,
    top_k: int,
    score_range: str,
):
    top_k = int(top_k)
    pool_size = min(max(top_k * POOL_MULTIPLIER, MIN_POOL_SIZE), MAX_POOL_SIZE)
    recommendations = build_pool(user_mood, activity, pool_size, artist_filter, genre_filter)
    filtered = filter_by_mood_match(recommendations, user_mood)
    filtered = filter_by_score_range(filtered, score_range)
    seed = secrets.randbelow(2**31 - 1)
    selected = sample_recommendations(filtered, top_k, seed=seed)
    if selected is None or selected.empty:
        message = (
            "<div class='warning-pill'>"
            "No hay canciones suficientes con esos filtros. Prueba otro rango o artista."
            "</div>"
        )
        query_state = {
            "mood": user_mood,
            "activity": activity,
            "artist_filter": artist_filter,
            "genre_filter": genre_filter,
            "score_range": score_range,
        }
        return message, None, recommendations, seed, [], pool_size, query_state
    cards_html, display_df = format_recommendations(selected)
    seen_ids = get_recommendation_ids(selected)
    query_state = {
        "mood": user_mood,
        "activity": activity,
        "artist_filter": artist_filter,
        "genre_filter": genre_filter,
        "score_range": score_range,
    }
    return cards_html, display_df, recommendations, seed, seen_ids, pool_size, query_state


def refresh_recommendations(
    top_k: int,
    score_range: str,
    artist_filter: str,
    genre_filter: str,
    pool,
    seed: int,
    seen_ids,
    pool_size: int,
    query_state,
):
    if not query_state:
        return (
            "<div class='warning-pill'>Ejecuta una busqueda primero.</div>",
            None,
            pool,
            seed,
            seen_ids,
            pool_size,
            query_state,
        )
    seen_ids = seen_ids or []
    top_k = int(top_k)
    score_range = score_range or query_state.get("score_range", "0.00 - 1.00")
    artist_filter = artist_filter or query_state.get("artist_filter", "")
    genre_filter = genre_filter or query_state.get("genre_filter", "Todos")
    available = filter_unseen(
        filter_by_score_range(filter_by_mood_match(pool, query_state["mood"]), score_range),
        seen_ids,
    )
    if available is None or len(available) < top_k:
        next_pool_size = min(max(pool_size * 2, MIN_POOL_SIZE), MAX_POOL_SIZE)
        if next_pool_size > (pool_size or 0):
            pool = build_pool(
                query_state["mood"],
                query_state["activity"],
                next_pool_size,
                artist_filter,
                genre_filter,
            )
            pool_size = next_pool_size
        available = filter_unseen(
            filter_by_score_range(filter_by_mood_match(pool, query_state["mood"]), score_range),
            seen_ids,
        )
    if available is None or len(available) == 0:
        return (
            "<div class='warning-pill'>No quedan opciones nuevas en ese rango. Cambia el rango o haz otra busqueda.</div>",
            None,
            pool,
            seed,
            seen_ids,
            pool_size,
            query_state,
        )
    new_seed = secrets.randbelow(2**31 - 1)
    selected = sample_recommendations(available, top_k, seed=new_seed)
    cards_html, display_df = format_recommendations(selected)
    new_seen = [*seen_ids, *get_recommendation_ids(selected)]
    return cards_html, display_df, pool, new_seed, new_seen, pool_size, query_state


def render_cards(recommendations, warning=None):
    cards = []
    for index, row in recommendations.iterrows():
        confidence = float(row.get("Confianza mood", 0.0))
        score_value = row.get("Score")
        score_label = "Score"
        score_display = "-"
        if score_value is not None and str(score_value).lower() != "nan":
            score_display = f"{float(score_value):.3f}"
        spotify_url = str(row.get("Spotify", ""))
        track_id = str(row.get("Track ID", "")).strip()
        spotify_link = ""
        if spotify_url.startswith("https://open.spotify.com/"):
            spotify_link = (
                f'<a class="spotify-link" href="{html.escape(spotify_url)}" '
                f'target="_blank" rel="noopener noreferrer">Abrir en Spotify</a>'
            )
        spotify_player = ""
        if track_id and track_id.lower() != "nan":
            safe_track_id = html.escape(track_id, quote=True)
            spotify_player = (
                '<iframe class="spotify-player" '
                f'src="https://open.spotify.com/embed/track/{safe_track_id}?utm_source=generator" '
                'title="Reproductor de Spotify" '
                'width="100%" height="80" frameborder="0" '
                'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
                'loading="lazy"></iframe>'
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
                <div class="song-actions">
                  {spotify_link}
                </div>
                {spotify_player}
              </div>
              <div class="score-box">
                <strong>{html.escape(score_display)}</strong>
                <span>{html.escape(score_label)}</span>
                <small>{confidence:.2f} mood</small>
              </div>
            </article>
            """
        )
    warning_html = ""
    if warning:
        warning_html = f'<div class="warning-pill">{html.escape(str(warning))}</div>'
    return f'{warning_html}<section class="cards-grid">{"".join(cards)}</section>'


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
.song-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.spotify-player {
  display: block;
  width: 100%;
  max-width: 520px;
  margin-top: 12px;
  border: 0;
  border-radius: 12px;
  background: #101418;
}
.warning-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid #3a4250;
  background: #1e242e;
  color: #f2c57c;
  padding: 8px 14px;
  border-radius: 999px;
  font-weight: 600;
  margin-bottom: 12px;
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
  max-width: 96px;
  font-size: 17px;
  line-height: 1;
  color: var(--accent-strong);
  text-align: center;
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
            artist_filter = gr.Textbox(
                value="",
                label="Filtrar por artista (opcional)",
                placeholder="Ejemplo: bad bunny, taylor swift...",
                scale=1,
            )
            genre_filter = gr.Dropdown(
                choices=GENRE_CHOICES,
                value="Todos",
                label="Filtrar por genero (opcional)",
                scale=1,
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Numero de canciones", scale=1)
            score_range = gr.Dropdown(
                choices=list(SCORE_RANGES.keys()),
                value="0.00 - 1.00",
                label="Rango de recomendacion",
                scale=1,
            )
        button = gr.Button("Buscar canciones", elem_classes=["primary-button"])
        refresh = gr.Button("Refrescar opciones")

    pool_state = gr.State(None)
    seed_state = gr.State(0)
    seen_state = gr.State([])
    pool_size_state = gr.State(0)
    query_state = gr.State(None)

    cards = gr.HTML(label="Recomendaciones")
    output = gr.Dataframe(label="Detalle tecnico", wrap=True)
    button.click(
        recommend_with_pool,
        inputs=[mood, activity, artist_filter, genre_filter, top_k, score_range],
        outputs=[cards, output, pool_state, seed_state, seen_state, pool_size_state, query_state],
    )
    refresh.click(
        refresh_recommendations,
        inputs=[top_k, score_range, artist_filter, genre_filter, pool_state, seed_state, seen_state, pool_size_state, query_state],
        outputs=[cards, output, pool_state, seed_state, seen_state, pool_size_state, query_state],
    )

    gr.Examples(
        examples=[
            ["triste", "quiero llorar", "", "Todos", 5],
            ["triste", "quiero fregar el suelo", "", "Todos", 5],
            ["triste", "voy a entrenar en el gym", "", "Todos", 5],
            ["feliz", "voy a estudiar programacion", "", "Todos", 5],
            ["tranquilo", "quiero dormir", "", "Todos", 5],
            ["energico", "voy a limpiar la casa", "", "Todos", 5],
        ],
        inputs=[mood, activity, artist_filter, genre_filter, top_k],
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base())
