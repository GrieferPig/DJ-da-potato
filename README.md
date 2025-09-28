# DJ da Potato

Automatic web radio station w/ smart transition

[Demo here](https://example.com)

## What it does

- Stream audio via a webpage (obviously)
- Smart transition based on music key & BPM* **
- Time-stretching BPM when needed

## Usage

If you're lazy - create a "music" subfolder and put your music there, filename should be formatted as below

> {track-number}{space}{some-random-thing}{space-dash-space}{artist}{space-dash-space}{track-title}

e.g. `05 Ponies at Dawn - ExplodingPonyToast - Dexter VS Sinister.mp3`

Analyze them once using

```bash
pip3 install -r requirements.txt
python3 library_manager.py
```

> **Note:** The web stack now runs on [eventlet](https://eventlet.net/) for cooperative concurrency. The dependency is already included in `requirements.txt`, but if you're deploying with Gunicorn remember to use the eventlet worker class, e.g. `gunicorn -k eventlet -w 1 app:app`.

Subsequent analyzes will skip the ones that are already analyzed. Neat™, huh?

Then launch the server using

```bash
python3 app.py
```

## it doesn't work on my machine 😭

What do you expect from a vibe-coded weekend project?

*Note: May not work on all songs, including orchestral, experimental electronic, or the one you just made in FL Studio in under 30 minutes and gave up.

*Disclaimer: "Smart" is for marketing only; actual results may vary.
