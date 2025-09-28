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

Then launch the server using

```bash
python3 app.py
```

## it doesn't work on my machine 😭

What do you expect from a vibe-coded weekend project?

<sub>* May not work on all songs, including: orchestral, experimental electronic, the one you just made in FL Studio in under 30 minutes and gave up</sub>

<sub>** "Smart" is for marketing only; vary by person</sub>
