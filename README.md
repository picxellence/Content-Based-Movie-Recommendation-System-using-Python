# Content-Based Movie Recommender 🎬

## Overview
Content-based recommendations using overview + genres + keywords + cast + director.

## Dataset
TMDB 5000 (put CSVs alongside the code; not committed).  

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Run (script)
python recommender.py

## Run (UI)
streamlit run app.py

## Project Structure
- recommender.py – build vectors + recommend()
- app.py – Streamlit UI
- requirements.txt – dependencies

## Future Work
- TF-IDF, posters via TMDB API, fuzzy search, hybrid with popularity.

## License
MIT

