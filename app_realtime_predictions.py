from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv

from src.realtime_predictions import (
    RealtimeMoodClassifier,
    extract_spotify_track_id,
    read_uploaded_table,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
CLASSIFIER = RealtimeMoodClassifier()

SOUNDNET_HOST = "track-analysis.p.rapidapi.com"
SOUNDNET_BASE_URL = "https://track-analysis.p.rapidapi.com"
SOUNDNET_KEY = os.getenv("RAPIDAPI_KEY", "46b2a3ca5dmshe7921879809bba9p189042jsn312046056254")
SOUNDNET_HEADERS = {
    "x-rapidapi-key": SOUNDNET_KEY,
    "x-rapidapi-host": SOUNDNET_HOST,
}
SOUNDNET_TIMEOUT_SECONDS = int(os.getenv("SOUNDNET_TIMEOUT_SECONDS", "300"))
SOUNDNET_SPOTIFY_ENDPOINT = os.getenv("SOUNDNET_SPOTIFY_ENDPOINT", "").strip()
SOUNDNET_FEATURE_ENDPOINT = os.getenv("SOUNDNET_FEATURE_ENDPOINT", "").strip()
SOUNDNET_SPOTIFY_ENDPOINTS = [
    "/pktx/spotify/{track_id}",
    "/spotify/{track_id}",
    "/v1/spotify/{track_id}",
    "/track/spotify/{track_id}",
]
SOUNDNET_FEATURE_ENDPOINTS = [
    "/pktx/spotify/{track_id}",
    "/spotify/{track_id}",
    "/audio-features/spotify/{track_id}",
    "/analysis/spotify/{track_id}",
]
SPOTIFY_OEMBED_URL = "https://open.spotify.com/oembed"
USE_SPOTIFY_OEMBED = os.getenv("USE_SPOTIFY_OEMBED", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
SPOTIFY_MARKET = os.getenv("SPOTIFY_MARKET", "").strip()
SPOTIFY_INCLUDE_GENRES = os.getenv("SPOTIFY_INCLUDE_GENRES", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_TRACK_URL = "https://api.spotify.com/v1/tracks/{track_id}"
SPOTIFY_ARTIST_URL = "https://api.spotify.com/v1/artists/{artist_id}"

_SPOTIFY_TOKEN_CACHE: dict[str, Any] = {"access_token": "", "expires_at": 0.0}


def resolve_soundnet_path(endpoints: Iterable[str] | str, track_id: str) -> str:
    if isinstance(endpoints, str):
        return endpoints.format(track_id=track_id)
    endpoint_list = list(endpoints)
    if not endpoint_list:
        raise ValueError("No hay endpoints configurados para SoundNet.")
    return endpoint_list[0].format(track_id=track_id)


def classify_manual(
    spotify_url,
    track_name,
    artists,
    track_genre,
    popularity,
    danceability,
    energy,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    loudness,
    tempo,
    duration_ms,
    spec_rate,
):
    frame = pd.DataFrame(
        [
            {
                "spotify_url": spotify_url,
                "track_name": track_name,
                "artists": artists,
                "track_genre": track_genre,
                "popularity": popularity,
                "danceability": danceability,
                "energy": energy,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "valence": valence,
                "loudness": loudness,
                "tempo": tempo,
                "duration_ms": duration_ms,
                "spec_rate": spec_rate,
            }
        ]
    )
    return classify_frame(frame)


def analyze_spotify_url(spotify_url: str):
    debug_info: dict[str, Any] = {"spotify_url": spotify_url}
    try:
        track_id = extract_spotify_track_id(spotify_url)
        debug_info["track_id"] = track_id
        features_payload, features_raw, feature_path = soundnet_get_by_spotify_id(
            track_id,
            SOUNDNET_FEATURE_ENDPOINT or SOUNDNET_FEATURE_ENDPOINTS,
            with_debug=True,
        )
        debug_info["soundnet_feature_endpoint"] = feature_path
        debug_info["features_payload_raw"] = features_raw
        features = build_features_from_soundnet(features_payload)
        debug_info["normalized_features"] = features
        if not features or features.get("id") is None:
            raise ValueError("No se pudieron obtener audio features del track.")
        spotify_payload: dict[str, Any] = {}
        spotify_artist_payload: dict[str, Any] = {}
        try:
            spotify_payload = fetch_spotify_track(track_id)
            debug_info["spotify_track_payload"] = spotify_payload
            if SPOTIFY_INCLUDE_GENRES:
                artist_id = None
                if isinstance(spotify_payload.get("artists"), list) and spotify_payload["artists"]:
                    artist_id = spotify_payload["artists"][0].get("id")
                if artist_id:
                    spotify_artist_payload = fetch_spotify_artist(str(artist_id))
                    debug_info["spotify_artist_payload"] = spotify_artist_payload
        except Exception as exc:
            debug_info["spotify_api_error"] = str(exc)

        oembed_payload: dict[str, Any] = {}
        if USE_SPOTIFY_OEMBED:
            try:
                oembed_payload = fetch_spotify_oembed(spotify_url)
            except Exception as exc:
                debug_info["spotify_oembed_error"] = str(exc)
            debug_info["spotify_oembed_payload"] = oembed_payload
        else:
            debug_info["spotify_oembed_skipped"] = True

        track_meta = normalize_track_payload(spotify_payload or oembed_payload, spotify_url)
        track_meta = enrich_track_meta(track_meta, spotify_payload, spotify_artist_payload)
        track_meta = enrich_track_meta(track_meta, oembed_payload, features_payload)
        debug_info["normalized_track_meta"] = track_meta
        track_genre = track_meta.get("track_genre") or "spotify_unknown"
        track_name = track_meta.get("track_name") or "Track sin nombre"
        artists = track_meta.get("artists") or "Artista desconocido"
        popularity = track_meta.get("popularity", 0)
        track_card = build_track_card(track_meta, spotify_url, track_genre)

        duration_ms = features.get("duration_ms") or track_meta.get("duration_ms") or 0
        speechiness = float(features.get("speechiness", 0.0))
        spec_rate = speechiness / max(float(duration_ms), 1.0)

        missing = []
        if track_name == "Track sin nombre":
            missing.append("track_name")
        if artists == "Artista desconocido":
            missing.append("artists")
        if track_genre == "spotify_unknown":
            missing.append("track_genre")
        debug_info["missing_fields"] = missing
        if missing:
            status = (
                "<div class='ok-box'><strong>Track analizado.</strong> "
                "Se han cargado automaticamente las features via SoundNet. "
                f"Faltan campos: {', '.join(missing)}.</div>"
            )
        else:
            status = (
                "<div class='ok-box'><strong>Track analizado.</strong> "
                "Se han cargado automaticamente las features via SoundNet.</div>"
            )
        return (
            status,
            track_card,
            track_name,
            artists,
            track_genre,
            popularity,
            round_feature(features.get("danceability", 0.0)),
            round_feature(features.get("energy", 0.0)),
            round_feature(speechiness),
            round_feature(features.get("acousticness", 0.0)),
            round_feature(features.get("instrumentalness", 0.0)),
            round_feature(features.get("liveness", 0.0)),
            round_feature(features.get("valence", 0.0)),
            round_numeric(features.get("loudness", 0.0)),
            round_numeric(features.get("tempo", 0.0)),
            int(duration_ms) if duration_ms else 0,
            round_feature(spec_rate),
            debug_info,
        )
    except Exception as exc:
        debug_info["error"] = str(exc)
        return (
            error_box(str(exc)),
            "",
            "",
            "",
            "",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            debug_info,
        )


def classify_dataset(file):
    if file is None:
        return error_box("Sube un archivo CSV o Parquet."), None, None
    try:
        frame = read_uploaded_table(file.name)
        return classify_frame(frame)
    except Exception as exc:
        return error_box(str(exc)), None, None


def classify_frame(frame: pd.DataFrame):
    try:
        classified, info = CLASSIFIER.classify_and_append(frame)
    except Exception as exc:
        return error_box(str(exc)), None, None

    display = classified[
        [
            col
            for col in [
                "track_id",
                "track_name",
                "artists",
                "track_genre",
                "predicted_mood",
                "mood_confidence",
                "proba_sad",
                "proba_happy",
                "proba_energetic",
                "proba_calm",
            ]
            if col in classified.columns
        ]
    ].copy()
    for col in ["mood_confidence", "proba_sad", "proba_happy", "proba_energetic", "proba_calm"]:
        if col in display.columns:
            display[col] = display[col].round(4)
    message = (
        f"<div class='ok-box'><strong>{len(display)} track(s) clasificados.</strong> "
        f"El catalogo de la app principal se ha actualizado en "
        f"<code>{html.escape(info['catalog_path'])}</code>.</div>"
    )
    return message, display, info


def error_box(message: str) -> str:
    return f"<div class='error-box'><strong>No se pudo clasificar.</strong> {html.escape(message)}</div>"


def ok_box(message: str) -> str:
    return f"<div class='ok-box'><strong>OK.</strong> {html.escape(message)}</div>"


def build_track_card(track: dict, spotify_url: str, genre: str) -> str:
    album = track.get("album", {}) if isinstance(track, dict) else {}
    if not isinstance(album, dict):
        album = {}
    images = album.get("images") or []
    cover_url = extract_cover_url(images)
    track_name = html.escape(str(track.get("name") or "Track sin nombre"))
    artists_value = track.get("artists") if isinstance(track, dict) else None
    artists_list = normalize_artists(artists_value)
    artists = html.escape(", ".join(artists_list) if artists_list else "Artista desconocido")
    album_name = html.escape(str(album.get("name") or "Album desconocido"))
    release_date = html.escape(str(album.get("release_date") or "-"))
    duration_ms = int(track.get("duration_ms") or 0)
    duration = format_duration(duration_ms)
    popularity = html.escape(str(track.get("popularity") or 0))
    genre = html.escape(str(genre or "spotify_unknown"))
    spotify_url = html.escape(str(spotify_url))

    cover_html = (
        f"<img class='track-cover' src='{cover_url}' alt='Portada del track' />"
        if cover_url
        else "<div class='track-cover placeholder'>Sin portada</div>"
    )
    return (
        "<section class='track-card'>"
        f"{cover_html}"
        "<div class='track-meta'>"
        f"<h3>{track_name}</h3>"
        f"<p class='track-artist'>{artists}</p>"
        "<div class='track-tags'>"
        f"<span>Album: {album_name}</span>"
        f"<span>Lanzamiento: {release_date}</span>"
        f"<span>Duracion: {duration}</span>"
        f"<span>Popularidad: {popularity}</span>"
        f"<span>Genero: {genre}</span>"
        "</div>"
        f"<a class='track-link' href='{spotify_url}' target='_blank' rel='noopener noreferrer'>"
        "Abrir en Spotify</a>"
        "</div>"
        "</section>"
    )


def format_duration(duration_ms: int) -> str:
    if duration_ms <= 0:
        return "-"
    total_seconds = int(round(duration_ms / 1000.0))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def soundnet_get_by_spotify_id(
    track_id: str,
    endpoints: Iterable[str] | str,
    with_debug: bool = False,
) -> Any:
    path = resolve_soundnet_path(endpoints, track_id)
    try:
        raw = soundnet_get_json(path)
    except ValueError as exc:
        raise ValueError(
            "SoundNet no devolvio datos para el track. "
            "Configura SOUNDNET_SPOTIFY_ENDPOINT/SOUNDNET_FEATURE_ENDPOINT con el endpoint exacto. "
            f"Ultimo error: {exc}"
        ) from exc
    normalized = normalize_soundnet_payload(raw)
    if with_debug:
        return normalized, raw, path
    return normalized


def soundnet_get_json(path: str) -> dict:
    if not SOUNDNET_KEY:
        raise ValueError("Falta RAPIDAPI_KEY para acceder a SoundNet.")
    url = f"{SOUNDNET_BASE_URL}{path}"
    response = requests.get(url, headers=SOUNDNET_HEADERS, timeout=SOUNDNET_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise ValueError(f"SoundNet API error {response.status_code}: {response.text}")
    return response.json()


def normalize_soundnet_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item:
                return item
        return payload[0] if payload else {}
    if isinstance(payload, str):
        return {"name": payload}
    return {}


def find_in_payload(payload: Any, keys: Iterable[str]) -> Any:
    key_set = {str(key).lower() for key in keys}
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in key_set and v not in (None, ""):
                return v
        for v in payload.values():
            found = find_in_payload(v, keys)
            if found not in (None, ""):
                return found
    if isinstance(payload, list):
        for item in payload:
            found = find_in_payload(item, keys)
            if found not in (None, ""):
                return found
    return None


def normalize_artists(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("artist")
            else:
                name = str(item)
            if name:
                names.append(name)
        return names
    if isinstance(value, dict):
        name = value.get("name") or value.get("artist")
        return [name] if name else []
    return [str(value)]


def extract_cover_url(value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "url",
            "image",
            "image_url",
            "cover",
            "cover_url",
            "thumbnail",
            "thumbnail_url",
        ):
            if value.get(key):
                return str(value[key])
    if isinstance(value, list):
        for item in value:
            url = extract_cover_url(item)
            if url:
                return url
    if isinstance(value, str):
        return value
    return ""


def normalize_duration_ms(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, str):
        text = value.strip()
        if ":" in text:
            parts = [part.strip() for part in text.split(":") if part.strip()]
            if all(part.isdigit() for part in parts):
                numbers = [int(part) for part in parts]
                if len(numbers) == 2:
                    minutes, seconds = numbers
                    total_seconds = minutes * 60 + seconds
                    return max(total_seconds, 0) * 1000
                if len(numbers) == 3:
                    hours, minutes, seconds = numbers
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    return max(total_seconds, 0) * 1000
    try:
        numeric = float(str(value).replace(",", "."))
    except ValueError:
        return 0
    if numeric <= 0:
        return 0
    if numeric < 1000:
        return int(numeric * 1000)
    return int(numeric)


def normalize_popularity(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        numeric = float(str(value).replace(",", "."))
    except ValueError:
        return 0
    if numeric < 0:
        return 0
    if 0 < numeric < 1:
        numeric = numeric * 100.0
    return int(round(min(numeric, 100.0)))


def fetch_spotify_oembed(spotify_url: str) -> dict:
    response = requests.get(SPOTIFY_OEMBED_URL, params={"url": spotify_url}, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"Spotify oEmbed error {response.status_code}: {response.text}")
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def get_spotify_access_token() -> str:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ValueError("Faltan SPOTIFY_CLIENT_ID o SPOTIFY_CLIENT_SECRET en el entorno.")
    now = time.time()
    cached = _SPOTIFY_TOKEN_CACHE.get("access_token")
    expires_at = float(_SPOTIFY_TOKEN_CACHE.get("expires_at", 0.0))
    if cached and now < expires_at - 30:
        return str(cached)
    response = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=10,
    )
    if response.status_code != 200:
        raise ValueError(f"Spotify token error {response.status_code}: {response.text}")
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise ValueError("Spotify token response sin access_token.")
    expires_in = int(payload.get("expires_in", 3600))
    _SPOTIFY_TOKEN_CACHE["access_token"] = token
    _SPOTIFY_TOKEN_CACHE["expires_at"] = now + expires_in
    return str(token)


def spotify_api_get(url: str, params: dict[str, Any] | None = None) -> dict:
    token = get_spotify_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, params=params or {}, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"Spotify API error {response.status_code}: {response.text}")
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def fetch_spotify_track(track_id: str) -> dict:
    url = SPOTIFY_TRACK_URL.format(track_id=track_id)
    params = {"market": SPOTIFY_MARKET} if SPOTIFY_MARKET else None
    return spotify_api_get(url, params=params)


def fetch_spotify_artist(artist_id: str) -> dict:
    url = SPOTIFY_ARTIST_URL.format(artist_id=artist_id)
    return spotify_api_get(url)


def normalize_track_payload(payload: Any, spotify_url: str) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    track_name = find_in_payload(
        payload,
        [
            "track_name",
            "trackname",
            "track_title",
            "tracktitle",
            "song",
            "song_name",
            "name",
            "title",
        ],
    ) or "Track sin nombre"
    artists_value = find_in_payload(
        payload,
        [
            "artists",
            "artist",
            "artist_name",
            "artistname",
            "artists_name",
            "artistnames",
            "author_name",
            "performer",
            "singer",
            "author",
        ],
    )
    artists = ", ".join(normalize_artists(artists_value)) or "Artista desconocido"
    genres_value = find_in_payload(payload, ["genres", "genre", "track_genre", "trackgenre", "style"])
    if isinstance(genres_value, list) and genres_value:
        track_genre = str(genres_value[0])
    elif genres_value:
        track_genre = str(genres_value)
    else:
        track_genre = "spotify_unknown"
    popularity_raw = find_in_payload(
        payload,
        ["popularity", "popularity_score", "popularityscore", "track_popularity", "score"],
    )
    popularity = normalize_popularity(popularity_raw)

    album_payload = find_in_payload(payload, ["album"]) if isinstance(payload, dict) else None
    album_name = "Album desconocido"
    release_date = "-"
    album_images = []
    if isinstance(album_payload, dict):
        album_name = album_payload.get("name") or album_payload.get("title") or album_name
        release_date = album_payload.get("release_date") or album_payload.get("release") or release_date
        album_images = album_payload.get("images") or []
    elif isinstance(album_payload, str) and album_payload:
        album_name = album_payload
    if album_name == "Album desconocido":
        album_name = (
            find_in_payload(payload, ["album_name", "albumname", "album_title", "albumtitle"])
            or album_name
        )
    if release_date == "-":
        release_date = find_in_payload(payload, ["release_date", "release", "releaseDate"]) or release_date
    cover_url = extract_cover_url(
        find_in_payload(payload, ["image", "image_url", "cover", "cover_url", "thumbnail", "images"])
        or album_images
    )
    duration_ms = normalize_duration_ms(find_in_payload(payload, ["duration_ms", "duration", "durationMs"]))

    return {
        "name": track_name,
        "track_name": track_name,
        "artists": artists,
        "track_genre": track_genre,
        "popularity": popularity,
        "duration_ms": duration_ms,
        "spotify_url": spotify_url,
        "album": {
            "name": album_name,
            "release_date": release_date,
            "images": [{"url": cover_url}] if cover_url else [],
        },
    }


def enrich_track_meta(track_meta: dict, *payloads: Any) -> dict:
    track_meta = dict(track_meta)
    for payload in payloads:
        if payload is None:
            continue
        if track_meta.get("track_name") in (None, "", "Track sin nombre"):
            name = find_in_payload(
                payload,
                [
                    "track_name",
                    "trackname",
                    "track_title",
                    "tracktitle",
                    "song",
                    "song_name",
                    "name",
                    "title",
                ],
            )
            if name:
                track_meta["track_name"] = str(name)
                track_meta["name"] = str(name)
        if track_meta.get("artists") in (None, "", "Artista desconocido"):
            artists_value = find_in_payload(
                payload,
                [
                    "artists",
                    "artist",
                    "artist_name",
                    "artistname",
                    "artists_name",
                    "artistnames",
                    "author_name",
                    "performer",
                    "singer",
                    "author",
                ],
            )
            artists = ", ".join(normalize_artists(artists_value))
            if artists:
                track_meta["artists"] = artists
        if track_meta.get("track_genre") in (None, "", "spotify_unknown"):
            genre_value = find_in_payload(
                payload,
                ["genres", "genre", "track_genre", "trackgenre", "style"],
            )
            if isinstance(genre_value, list) and genre_value:
                track_meta["track_genre"] = str(genre_value[0])
            elif genre_value:
                track_meta["track_genre"] = str(genre_value)
        if not track_meta.get("duration_ms"):
            duration = normalize_duration_ms(
                find_in_payload(payload, ["duration_ms", "duration", "durationMs"])
            )
            if duration:
                track_meta["duration_ms"] = duration
        if not track_meta.get("popularity"):
            popularity_raw = find_in_payload(
                payload,
                ["popularity", "popularity_score", "popularityscore", "track_popularity", "score"],
            )
            if popularity_raw is not None:
                track_meta["popularity"] = normalize_popularity(popularity_raw)
    return track_meta


def scale_unit_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        numeric = float(str(value).replace(",", "."))
    except ValueError:
        return 0.0
    if numeric < 0:
        return 0.0
    if numeric > 1.0:
        numeric = numeric / 100.0
    return min(max(numeric, 0.0), 1.0)


def parse_loudness(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, str):
        cleaned = value.lower().replace("db", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    try:
        numeric = float(value)
    except ValueError:
        return 0.0
    if 0.0 <= numeric <= 1.0:
        numeric = numeric * 100.0
    if 0.0 <= numeric <= 100.0:
        return (numeric / 100.0) * 60.0 - 60.0
    return numeric


def build_features_from_soundnet(payload: Any) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    danceability = scale_unit_value(find_in_payload(payload, ["danceability"]))
    energy = scale_unit_value(find_in_payload(payload, ["energy"]))
    acousticness = scale_unit_value(find_in_payload(payload, ["acousticness"]))
    speechiness = scale_unit_value(find_in_payload(payload, ["speechiness"]))
    instrumentalness = scale_unit_value(find_in_payload(payload, ["instrumentalness"]))
    liveness = scale_unit_value(find_in_payload(payload, ["liveness"]))
    happiness = find_in_payload(payload, ["happiness", "valence"])
    valence = scale_unit_value(happiness)
    tempo = find_in_payload(payload, ["tempo", "bpm"]) or 0.0
    try:
        tempo = float(str(tempo).replace(",", "."))
    except ValueError:
        tempo = 0.0
    loudness = parse_loudness(find_in_payload(payload, ["loudness", "loudness_db", "loudnessdb"]))
    duration_ms = normalize_duration_ms(find_in_payload(payload, ["duration_ms", "duration", "durationMs"]))

    return {
        "id": find_in_payload(payload, ["id", "track_id"]) or "",
        "danceability": danceability,
        "energy": energy,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "loudness": loudness,
        "tempo": tempo,
        "duration_ms": duration_ms,
    }


def round_feature(value: Any, decimals: int = 3) -> float:
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return 0.0


def round_numeric(value: Any, decimals: int = 3) -> float:
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return 0.0



CSS = """
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap");

.gradio-container {
  max-width: 1320px !important;
  margin: 0 auto !important;
  font-family: "Space Grotesk", "Trebuchet MS", sans-serif !important;
}
body, .gradio-container {
  background: #101418 !important;
}
.top-band {
  padding: 28px 30px;
  border: 1px solid #2b313c;
  background: #171c22;
  border-radius: 10px;
  color: #f4f0e8;
  margin-bottom: 18px;
}
.top-band h1 {
  margin: 0 0 8px;
  font-size: 34px;
  line-height: 1.1;
}
.top-band p {
  margin: 0;
  max-width: 820px;
  color: #c7ccd6;
  font-size: 16px;
}
.primary-button button {
  min-height: 48px !important;
  font-weight: 800 !important;
  background: #f2b361 !important;
  color: #1c1407 !important;
  border: 0 !important;
}
.track-card {
    display: grid;
    grid-template-columns: 120px 1fr;
    gap: 16px;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #2b313c;
    background: #141820;
    margin-bottom: 12px;
}
.track-cover {
    width: 120px;
    height: 120px;
    border-radius: 8px;
    object-fit: cover;
    border: 1px solid #2b313c;
}
.track-cover.placeholder {
    display: grid;
    place-items: center;
    color: #b6beca;
    background: #1a1f27;
    font-size: 12px;
}
.track-meta h3 {
    margin: 0 0 6px;
    font-size: 20px;
}
.track-artist {
    margin: 0 0 8px;
    color: #c7ccd6;
}
.track-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}
.track-tags span {
    border: 1px solid #2f3846;
    background: #1b212b;
    color: #cdd4df;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 12px;
}
.track-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 32px;
    padding: 0 12px;
    border-radius: 999px;
    background: #1db954;
    color: #07130b !important;
    font-weight: 700;
    text-decoration: none !important;
}
.ok-box,
.error-box {
  border-radius: 8px;
  padding: 12px 14px;
  margin: 8px 0;
}
.ok-box {
  border: 1px solid #2f6b47;
  background: #12251a;
  color: #d9f7e4;
}
.error-box {
  border: 1px solid #7d3b3b;
  background: #271616;
  color: #ffd8d8;
}
code {
  color: #f2c57c;
}
"""


with gr.Blocks(title="Clasificador realtime de tracks") as demo:
    gr.HTML(
        """
        <section class="top-band">
          <h1>Clasificador realtime de tracks</h1>
          <p>Introduce una cancion manualmente o sube un dataset CSV/Parquet. Las predicciones se anexan al catalogo que consume la app principal de recomendaciones.</p>
        </section>
        """
    )
    gr.Markdown(
        "Columnas minimas para dataset: "
        + ", ".join(f"`{col}`" for col in CLASSIFIER.feature_cols)
        + ". Tambien es obligatoria `spotify_url`. Metadatos opcionales: `track_name`, `artists`, `track_genre`, `popularity`."
    )

    with gr.Tab("Track individual"):
        spotify_url = gr.Textbox(
            value="https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
            label="URL Spotify del track",
            placeholder="https://open.spotify.com/track/...",
        )
        analyze_button = gr.Button("Analizar URL", elem_classes=["primary-button"])
        with gr.Row():
            track_name = gr.Textbox(value="Demo track", label="Track", interactive=False)
            artists = gr.Textbox(value="Demo artist", label="Artista", interactive=False)
            track_genre = gr.Textbox(value="manual_input", label="Genero", interactive=False)
            popularity = gr.Number(value=50, label="Popularidad", interactive=False)
        with gr.Row():
            danceability = gr.Slider(0, 1, value=0.65, step=0.01, label="danceability", interactive=False)
            energy = gr.Slider(0, 1, value=0.72, step=0.01, label="energy", interactive=False)
            speechiness = gr.Slider(0, 1, value=0.08, step=0.01, label="speechiness", interactive=False)
            acousticness = gr.Slider(0, 1, value=0.22, step=0.01, label="acousticness", interactive=False)
        with gr.Row():
            instrumentalness = gr.Slider(0, 1, value=0.02, step=0.01, label="instrumentalness", interactive=False)
            liveness = gr.Slider(0, 1, value=0.15, step=0.01, label="liveness", interactive=False)
            valence = gr.Slider(0, 1, value=0.70, step=0.01, label="valence", interactive=False)
        with gr.Row():
            loudness = gr.Number(value=-7.5, label="loudness", interactive=False)
            tempo = gr.Number(value=124, label="tempo", interactive=False)
            duration_ms = gr.Number(value=210000, label="duration_ms", interactive=False)
        spec_rate = gr.Number(value=0.0, label="spec_rate", visible=False)
        manual_button = gr.Button("Clasificar y anadir al catalogo", elem_classes=["primary-button"])

    with gr.Tab("Dataset"):
        dataset_file = gr.File(label="CSV o Parquet", file_types=[".csv", ".parquet", ".pq"])
        dataset_button = gr.Button("Clasificar dataset y anadir al catalogo", elem_classes=["primary-button"])

    status = gr.HTML()
    track_card = gr.HTML()
    predictions = gr.Dataframe(label="Predicciones", wrap=True)
    technical_info = gr.JSON(label="Detalle tecnico")

    analyze_button.click(
        analyze_spotify_url,
        inputs=[spotify_url],
        outputs=[
            status,
            track_card,
            track_name,
            artists,
            track_genre,
            popularity,
            danceability,
            energy,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            loudness,
            tempo,
            duration_ms,
            spec_rate,
            technical_info,
        ],
    )

    manual_button.click(
        classify_manual,
        inputs=[
            spotify_url,
            track_name,
            artists,
            track_genre,
            popularity,
            danceability,
            energy,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            loudness,
            tempo,
            duration_ms,
            spec_rate,
        ],
        outputs=[status, predictions, technical_info],
    )
    dataset_button.click(
        classify_dataset,
        inputs=[dataset_file],
        outputs=[status, predictions, technical_info],
    )

    gr.Examples(
        examples=[
            [
                "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
                "Cancion calmada",
                "Demo",
                "ambient",
                35,
                0.32,
                0.18,
                0.04,
                0.82,
                0.30,
                0.10,
                0.28,
                -18.0,
                78,
                240000,
                0.0,
            ],
            [
                "https://open.spotify.com/track/7ouMYWpwJ422jRcDASZB7P",
                "Cancion gym",
                "Demo",
                "dance",
                70,
                0.82,
                0.90,
                0.07,
                0.06,
                0.01,
                0.18,
                0.78,
                -5.5,
                132,
                198000,
                0.0,
            ],
        ],
        inputs=[
            spotify_url,
            track_name,
            artists,
            track_genre,
            popularity,
            danceability,
            energy,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            loudness,
            tempo,
            duration_ms,
            spec_rate,
        ],
    )

if __name__ == "__main__":
    launch_kwargs = {
        "server_port": 7861,
        "css": CSS,
        "theme": gr.themes.Base(),
    }
    ssl_certfile = os.getenv("SSL_CERTFILE", "").strip()
    ssl_keyfile = os.getenv("SSL_KEYFILE", "").strip()
    if ssl_certfile and ssl_keyfile:
        launch_kwargs["ssl_certfile"] = ssl_certfile
        launch_kwargs["ssl_keyfile"] = ssl_keyfile
    demo.launch(**launch_kwargs)
