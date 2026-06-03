# RGDA Web Presentation

Interactive Vite presentation for the Data Assimilation + Machine Learning final project.

## Run

```bash
npm install
npm run dev -- --port 5173
```

Open `http://127.0.0.1:5173/`.

## Deployed site

GitHub Pages URL:

```text
https://dyu056.github.io/da_based_video_prediction/
```

## Controls

- Left/right arrows: change slide
- Space: next slide
- Qualitative slide: use the dropdown to switch cases
- Qualitative slide: `Play all` starts retrieved / baseline / scene-assimilated videos together
- Number keys `1` to `6`: jump between qualitative cases

## Media

All presentation assets live under `public/media`.
Each qualitative case contains:

- `anchor.png`
- `reference.mp4`
- `baseline.mp4`
- `ours.mp4`
