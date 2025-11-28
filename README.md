# NeuroMotion: Advanced Parkinson's Disease Risk Assessment Tool

NeuroMotion is a comprehensive screening platform for assessing Parkinsonian risk using evidence-based digital motor and symptom tests. It combines a modern React frontend (Vite) with a FastAPI backend for real-time analysis and analytics.

---

## Features

- **Symptom Survey**: Questionnaire based on UPDRS criteria to assess key Parkinsonian symptoms.
- **Finger Tapping Test**: Measures motor speed and rhythm consistency through rapid alternating finger taps.
- **Spiral Drawing Test**: Analyzes fine motor control and tremor patterns via a guided spiral drawing task.
- **Automated Risk Scoring**: Aggregates results from all tests to provide an overall risk score and clinical recommendation.
- **Results Dashboard**: Visualizes assessment results, risk levels, and historical analytics.
- **Session History**: Tracks all previous assessments with detailed breakdowns and trends.

---

## Tech Stack

- **Frontend**: React (Vite, Tailwind CSS, Recharts)
- **Backend**: FastAPI (Python), OpenCV, SQLite
- **Communication**: REST API (JSON)

---

## Project Structure

```
Backend/           # FastAPI backend (main.py, requirements.txt)
pk-hack/           # Frontend (React + Vite)
  src/             # React source code
    components/    # UI and test components
    App.jsx        # Main app logic
    ...
  public/          # Static assets
  package.json     # Frontend dependencies
  requirements.txt # Backend dependencies
  main.py          # FastAPI backend (duplicate for dev)
```

---

## Quick Start

### 1. Install Dependencies

From the `pk-hack` directory:

```bash
# Install Python backend dependencies
pip install -r requirements.txt

# Install Node frontend dependencies
npm install
```

Or run the all-in-one script (Windows):

```bash
setup-and-run.cmd
```

### 2. Start the Application

- **Backend**: `uvicorn main:app --reload --port 8000`
- **Frontend**: `npm run dev` (default: http://localhost:5173)

Or use the provided scripts:

```bash
run-dev.cmd         # Start frontend only
setup-and-run.cmd   # Setup and start both servers
```

### 3. Access the App

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## How It Works

1. **Survey**: User answers questions about tremor, rigidity, bradykinesia, balance, and walking. Each answer is weighted and scored.
2. **Tapping Test**: User taps a button as quickly and consistently as possible for 10 seconds. Speed and rhythm are analyzed.
3. **Spiral Drawing**: User draws a spiral on screen. The backend analyzes smoothness and tremor using OpenCV.
4. **Aggregation**: Scores from all tests are combined (weighted average) to produce an overall risk score and recommendation.
5. **Results & Analytics**: Results are visualized, stored, and can be reviewed in the dashboard/history.

---

## API Endpoints (Backend)

- `POST /api/v1/analyze/survey`   — Analyze survey answers
- `POST /api/v1/analyze/taps`     — Analyze tapping intervals
- `POST /api/v1/analyze/spiral`   — Analyze spiral drawing (image upload)
- `POST /api/v1/aggregate`        — Aggregate all test scores
- `GET  /api/v1/sessions`         — List all sessions
- `GET  /api/v1/statistics`       — Get analytics/statistics

---

## Customization & Development

- **Frontend**: Edit React components in `pk-hack/src/components/`.
- **Backend**: Edit FastAPI logic in `Backend/main.py`.
- **Styling**: Tailwind CSS (`tailwind.config.cjs`, `index.css`).
- **Database**: SQLite (`test_sessions.db`), auto-initialized.

---

## Requirements

- Python 3.8+
- Node.js 16+
- See `requirements.txt` and `package.json` for full dependencies.

---

## Disclaimer

This tool is for screening and educational purposes only. It does **not** provide a medical diagnosis. Please consult a qualified healthcare professional for clinical evaluation and advice.

---

## References
- [Parkinson's Foundation](https://www.parkinson.org)
- [Michael J. Fox Foundation](https://www.michaeljfox.org)

---

## License
MIT
