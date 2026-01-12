# Qualtrics Latency/Streaming Study — Local Mock

This folder provides a self-contained HTML/JS harness to test the Qualtrics study behavior locally: streaming vs non-streaming, randomized trials, timing capture, post-trial sliders, and wrap-up. It mirrors the logic of the Qualtrics code you pasted earlier.

## Files

- `index.html`: Single-page app container
- `styles.css`: Minimal styling
- `app.js`: Logic for trials, streaming, data capture, and export

## Run

Open `index.html` in a browser, or serve the directory:

```bash
# Option A: Python
python3 -m http.server 8080 --directory /workspace/qualtrics-mock

# Option B: Node (if you prefer)
npx http-server /workspace/qualtrics-mock -p 8080 --silent
```

Then visit `http://localhost:8080`.

## What to verify

- Non-streaming trials (A–C) show a blank state until full answer pops at ~0.5s/2s/6s.
- Streaming trials (D–E) begin showing text ~0.3s then complete at ~2s/6s.
- The debug panel shows measured first-token and full-render times.
- You can complete the matrix, sliders, and wrap-up, then download a JSON summary.

This harness is only for local validation. The Qualtrics implementation uses the same timing rules and content, with per-trial Embedded Data stored in Survey Flow.