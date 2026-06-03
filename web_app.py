from __future__ import annotations

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import app as recommender_app
import app_realtime_predictions as realtime_app

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

AUTH_COOKIE = "mma_logged_in"


app = FastAPI(title="Music Mood Activity Web")


def is_public_path(path: str) -> bool:
    return (
        path == "/"
        or path == "/health"
        or path.startswith("/assets/")
        or path.startswith("/login")
        or path.startswith("/favicon")
    )


@app.middleware("http")
async def browser_auth_gate(request: Request, call_next):
    path = request.url.path
    if not is_public_path(path) and request.cookies.get(AUTH_COOKIE) != "1":
        return RedirectResponse("/", status_code=302)
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def login_page() -> str:
    return page_shell(
        """
        <main class="login-shell">
          <section class="login-panel">
            <p class="eyebrow">Music Mood Activity</p>
            <h1>Acceso al laboratorio musical</h1>
            <p class="intro">Registra un usuario local o inicia sesion para abrir el recomendador y el clasificador realtime.</p>

            <form id="auth-form" class="auth-form">
              <label>
                Usuario
                <input id="username" autocomplete="username" minlength="3" required placeholder="tu usuario" />
              </label>
              <label>
                Password
                <input id="password" type="password" autocomplete="current-password" minlength="4" required placeholder="password local" />
              </label>
              <div class="actions">
                <button type="submit" data-mode="login">Entrar</button>
                <button type="button" class="secondary" id="register">Registrarme</button>
              </div>
              <p id="message" class="message"></p>
            </form>
          </section>
        </main>
        <script>
          const form = document.querySelector("#auth-form");
          const userInput = document.querySelector("#username");
          const passInput = document.querySelector("#password");
          const registerButton = document.querySelector("#register");
          const message = document.querySelector("#message");

          function keyFor(user) {
            return "mma_user_" + user.trim().toLowerCase();
          }

          function enter() {
            localStorage.setItem("mma_current_user", userInput.value.trim());
            document.cookie = "mma_logged_in=1; path=/; max-age=2592000; SameSite=Lax";
            window.location.href = "/app";
          }

          registerButton.addEventListener("click", () => {
            if (!form.reportValidity()) return;
            localStorage.setItem(keyFor(userInput.value), passInput.value);
            message.textContent = "Usuario registrado en este navegador.";
            enter();
          });

          form.addEventListener("submit", (event) => {
            event.preventDefault();
            if (!form.reportValidity()) return;
            const stored = localStorage.getItem(keyFor(userInput.value));
            if (!stored) {
              message.textContent = "Ese usuario no existe en este navegador. Registralo primero.";
              return;
            }
            if (stored !== passInput.value) {
              message.textContent = "Password incorrecta.";
              return;
            }
            enter();
          });
        </script>
        """,
        title="Login",
    )


@app.get("/app", response_class=HTMLResponse)
def recommender_page() -> str:
    return framed_page(
        title="Recomendador",
        active="app",
        iframe_src="/gradio/recommender/",
    )


@app.get("/realtime", response_class=HTMLResponse)
def realtime_page() -> str:
    return framed_page(
        title="Realtime",
        active="realtime",
        iframe_src="/gradio/realtime/",
    )


@app.get("/logout", response_class=HTMLResponse)
def logout_page() -> str:
    return page_shell(
        """
        <main class="login-shell">
          <section class="login-panel compact">
            <p class="eyebrow">Sesion cerrada</p>
            <h1>Hasta la proxima</h1>
            <p class="intro">Se ha cerrado la sesion de este navegador.</p>
            <a class="button-link" href="/">Volver al login</a>
          </section>
        </main>
        <script>
          document.cookie = "mma_logged_in=; path=/; max-age=0; SameSite=Lax";
          localStorage.removeItem("mma_current_user");
        </script>
        """,
        title="Salir",
    )


def framed_page(title: str, active: str, iframe_src: str) -> str:
    app_class = "active" if active == "app" else ""
    realtime_class = "active" if active == "realtime" else ""
    return page_shell(
        f"""
        <div class="app-layout">
          <header class="top-nav">
            <a class="brand" href="/app">Music Mood Activity</a>
            <nav>
              <a class="{app_class}" href="/app">Recomendador</a>
              <a class="{realtime_class}" href="/realtime">Realtime</a>
              <a href="/logout">Salir</a>
            </nav>
          </header>
          <iframe class="gradio-frame" src="{iframe_src}" title="{title}"></iframe>
        </div>
        """,
        title=title,
    )


def page_shell(body: str, title: str) -> str:
    return f"""
    <!doctype html>
    <html lang="es">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title} | Music Mood Activity</title>
        <style>
          :root {{
            color-scheme: dark;
            --bg: #0b0f14;
            --panel: #151a22;
            --panel-strong: #1e2631;
            --text: #f7f3ea;
            --muted: #b9c0cc;
            --line: #2b313c;
            --accent: #f2b361;
            --accent-text: #1c1407;
            --green: #1db954;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
              radial-gradient(circle at 18% 12%, rgba(242, 179, 97, 0.13), transparent 28%),
              linear-gradient(180deg, #0b0f14 0%, #101521 100%);
            color: var(--text);
          }}
          a {{ color: inherit; text-decoration: none; }}
          .login-shell {{
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 28px;
          }}
          .login-panel {{
            width: min(440px, 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(21, 26, 34, 0.92);
            padding: 30px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
          }}
          .login-panel.compact {{ text-align: center; }}
          .eyebrow {{
            margin: 0 0 10px;
            color: var(--accent);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
          }}
          h1 {{
            margin: 0;
            font-size: clamp(30px, 7vw, 46px);
            line-height: 1.04;
            letter-spacing: 0;
          }}
          .intro {{
            margin: 16px 0 24px;
            color: var(--muted);
            line-height: 1.55;
          }}
          .auth-form {{ display: grid; gap: 16px; }}
          label {{
            display: grid;
            gap: 8px;
            color: #dfe4ec;
            font-size: 14px;
            font-weight: 700;
          }}
          input {{
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #0f141b;
            color: var(--text);
            padding: 13px 14px;
            font: inherit;
          }}
          input:focus {{
            outline: 2px solid rgba(242, 179, 97, 0.55);
            border-color: var(--accent);
          }}
          .actions {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
          }}
          button,
          .button-link {{
            display: inline-flex;
            justify-content: center;
            align-items: center;
            min-height: 44px;
            border: 1px solid var(--accent);
            border-radius: 8px;
            background: var(--accent);
            color: var(--accent-text);
            padding: 0 16px;
            font: inherit;
            font-weight: 800;
            cursor: pointer;
          }}
          button.secondary {{
            border-color: var(--line);
            background: var(--panel-strong);
            color: var(--text);
          }}
          .message {{
            min-height: 20px;
            margin: 0;
            color: var(--accent);
            font-size: 13px;
          }}
          .app-layout {{
            min-height: 100vh;
            display: grid;
            grid-template-rows: 64px 1fr;
          }}
          .top-nav {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 0 22px;
            border-bottom: 1px solid var(--line);
            background: rgba(11, 15, 20, 0.94);
            backdrop-filter: blur(16px);
          }}
          .brand {{
            font-weight: 900;
            color: var(--text);
            white-space: nowrap;
          }}
          nav {{
            display: flex;
            align-items: center;
            gap: 8px;
            overflow-x: auto;
          }}
          nav a {{
            border: 1px solid transparent;
            border-radius: 8px;
            color: var(--muted);
            padding: 9px 12px;
            font-weight: 800;
            white-space: nowrap;
          }}
          nav a.active {{
            border-color: rgba(242, 179, 97, 0.45);
            background: rgba(242, 179, 97, 0.12);
            color: var(--accent);
          }}
          .gradio-frame {{
            width: 100%;
            height: calc(100vh - 64px);
            border: 0;
            background: var(--bg);
          }}
          @media (max-width: 620px) {{
            .top-nav {{
              height: auto;
              min-height: 74px;
              align-items: flex-start;
              flex-direction: column;
              padding: 12px;
            }}
            .app-layout {{ grid-template-rows: auto 1fr; }}
            .gradio-frame {{ height: calc(100vh - 106px); }}
            .actions {{ grid-template-columns: 1fr; }}
          }}
        </style>
      </head>
      <body>{body}</body>
    </html>
    """


app = gr.mount_gradio_app(
    app,
    recommender_app.demo,
    path="/gradio/recommender",
    css=recommender_app.CSS,
    theme=gr.themes.Base(),
)
app = gr.mount_gradio_app(
    app,
    realtime_app.demo,
    path="/gradio/realtime",
    css=realtime_app.CSS,
    theme=gr.themes.Base(),
)
