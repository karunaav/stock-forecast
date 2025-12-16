"""
RNN Stock Forecast Dashboard (FastAPI + Dash)
- UI intentionally kept simple and contains NO mention of any external author's name.
- Model: SimpleRNN-based time-series forecaster inspired by typical RNN stock forecasting workflows.
Run:
  pip install -r requirements.txt
  uvicorn app:app --port 8000 --reload
Then open http://127.0.0.1:8000/
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF logs (doesn't disable oneDNN)

from datetime import date
from typing import Optional

from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from src.rnn_model import train_and_forecast, ForecastResult


APP_TITLE = "RNN Stock Forecast"

DEFAULT_TICKER = "AAPL"
DEFAULT_START = date(2020, 1, 1)
DEFAULT_END = date.today()

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"]


def stat_card(title: str, value: str, subtitle: str = ""):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="stat-title"),
                html.Div(value, className="stat-value"),
                html.Div(subtitle, className="stat-subtitle"),
            ]
        ),
        className="stat-card",
    )


dash_app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title=APP_TITLE,
    suppress_callback_exceptions=True,
)
server = dash_app.server  # Flask WSGI app (used by FastAPI mount)


dash_app.layout = dbc.Container(
    [
        html.H2(APP_TITLE, className="mt-4 mb-2"),
        html.Div(
            "Predict next-day closing price using a simple RNN trained on historical closing prices.",
            className="text-muted mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Ticker"),
                        dcc.Dropdown(
                            id="ticker",
                            options=[{"label": t, "value": t} for t in TICKERS],
                            value=DEFAULT_TICKER,
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Date range"),
                        dcc.DatePickerRange(
                            id="date_range",
                            start_date=DEFAULT_START,
                            end_date=DEFAULT_END,
                            display_format="YYYY-MM-DD",
                            min_date_allowed=date(2000, 1, 1),
                            max_date_allowed=DEFAULT_END,
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Label("Lookback (days)"),
                        dbc.Input(id="lookback", type="number", min=10, max=200, step=1, value=60),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("Epochs"),
                        dbc.Input(id="epochs", type="number", min=1, max=200, step=1, value=10),
                    ],
                    md=2,
                ),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Forecast",
                        id="run_btn",
                        color="primary",
                        className="mt-3",
                        n_clicks=0,
                    ),
                    md="auto",
                ),
                dbc.Col(
                    html.Div(
                        id="status",
                        className="mt-3 text-muted",
                        style={"whiteSpace": "pre-wrap"},
                    )
                ),
            ],
            className="align-items-center",
        ),
        html.Hr(className="my-4"),
        dbc.Row(
            [
                dbc.Col(stat_card("Current close", "—", "From most recent row"), md=6, id="card_current"),
                dbc.Col(stat_card("Predicted next close", "—", "Model output"), md=6, id="card_pred"),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        dcc.Graph(id="price_graph", figure={"data": [], "layout": {"height": 520}}),
                        type="default",
                    ),
                    md=12,
                    className="mt-3",
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        id="notes",
                        className="text-muted mt-3",
                        style={"whiteSpace": "pre-wrap"},
                    )
                )
            ]
        ),
    ],
    fluid=True,
)


@dash_app.callback(
    Output("price_graph", "figure"),
    Output("card_current", "children"),
    Output("card_pred", "children"),
    Output("status", "children"),
    Output("notes", "children"),
    Input("run_btn", "n_clicks"),
    State("ticker", "value"),
    State("date_range", "start_date"),
    State("date_range", "end_date"),
    State("lookback", "value"),
    State("epochs", "value"),
    prevent_initial_call=True,
)
def run_forecast(n_clicks: int, ticker: str, start_date: str, end_date: str, lookback: int, epochs: int):
    try:
        result: ForecastResult = train_and_forecast(
            ticker=ticker,
            start=start_date,
            end=end_date,
            lookback=int(lookback),
            epochs=int(epochs),
        )
    except Exception as e:
        msg = f"Error: {type(e).__name__}: {e}"
        empty_fig = {"data": [], "layout": {"title": "No data", "height": 520}}
        return (
            empty_fig,
            stat_card("Current close", "—"),
            stat_card("Predicted next close", "—"),
            msg,
            "Tip: make sure you have an internet connection (yfinance downloads data).",
        )

    # Build Plotly figure
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.dates, y=result.actual_close, mode="lines", name="Actual Close"))
    fig.add_trace(go.Scatter(x=result.dates, y=result.pred_close, mode="lines", name="Predicted Close"))
    fig.add_trace(
        go.Scatter(
            x=[result.next_date],
            y=[result.pred_next_close],
            mode="markers",
            name="Predicted Next Close",
        )
    )
    fig.update_layout(
        title=f"{ticker}: Actual vs Predicted Close",
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    card_current = stat_card("Current close", f"${result.current_close:,.2f}", "From most recent row")
    card_pred = stat_card("Predicted next close", f"${result.pred_next_close:,.2f}", f"Next trading day: {result.next_date}")

    status = (
        f"Done.\n"
        f"Samples: {result.n_rows} | Train: {result.train_size} | Test: {result.test_size}\n"
        f"Lookback: {result.lookback} | Epochs: {result.epochs}"
    )

    notes = (
        "Notes:\n"
        "- This is a learning-style RNN forecaster and is NOT financial advice.\n"
        "- Results can vary a lot depending on date range, lookback, and epochs.\n"
        "- If forecasting feels slow, reduce epochs (e.g., 5–10) or shorten the date range."
    )

    return fig, card_current, card_pred, status, notes


# ---- FastAPI wrapper (so you can run with uvicorn app:app) ----
fastapi_app = FastAPI(title=APP_TITLE)

@fastapi_app.get("/health")
def health():
    return {"status": "ok"}

# Mount Dash (Flask) on root.
fastapi_app.mount("/", WSGIMiddleware(server))

# Uvicorn expects variable named "app"
app = fastapi_app
