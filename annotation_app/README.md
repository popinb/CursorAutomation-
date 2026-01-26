# 🏷️ Annotation App

A modern, user-friendly text annotation tool built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## Features

- **Clean Modern UI** - Beautiful, intuitive interface for efficient annotation
- **Flexible Labels** - Add, remove, and customize annotation labels
- **Progress Tracking** - Visual progress bar and statistics
- **Data Import** - Support for CSV and JSON file formats
- **Auto-Save** - Save annotations to JSON for later use
- **Export** - Download annotated data as JSON
- **Resume Work** - Load existing projects to continue annotation
- **Label Distribution** - Visual chart showing annotation distribution

## Quick Start

### 1. Install Dependencies

```bash
cd annotation_app
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Start Annotating

1. **Load Data**: Upload a CSV/JSON file or click "Load Sample Data"
2. **Configure Labels**: Use the sidebar to add/remove labels
3. **Annotate**: Click on label buttons to annotate each item
4. **Navigate**: Use Previous/Next or jump to specific items
5. **Export**: Save your annotations when done

## Data Formats

### CSV Format
First column should contain the text to annotate:

```csv
text,category
"The product is great!",review
"Terrible experience.",review
```

### JSON Format
Array of strings or objects with 'text' field:

```json
[
    {"text": "The product is great!"},
    {"text": "Terrible experience."}
]
```

Or simply:
```json
[
    "The product is great!",
    "Terrible experience."
]
```

## Project Structure

```
annotation_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── data/              # Data directory
    ├── sample_data.json
    ├── sample_data.csv
    └── *_annotations.json  # Saved annotations
```

## Export Format

Annotations are saved in JSON format:

```json
{
    "project": "my_project",
    "total_items": 100,
    "annotated_items": 50,
    "labels": ["Positive", "Negative", "Neutral"],
    "annotations": [
        {"text": "Great product!", "label": "Positive"},
        {"text": "Bad experience", "label": "Negative"}
    ]
}
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Navigate | Use browser tab + enter |

## Use Cases

- **Sentiment Analysis** - Label text as positive/negative/neutral
- **Intent Classification** - Categorize user queries by intent
- **Topic Labeling** - Assign topics to documents
- **Quality Assessment** - Rate content quality
- **Named Entity Recognition** - Mark entity types (with custom labels)

## Customization

You can easily customize the app by modifying `app.py`:

- **Default Labels**: Change the initial labels in `init_session_state()`
- **Colors**: Modify the CSS in the `st.markdown()` style block
- **Layout**: Adjust column ratios and component arrangement

## License

MIT License - feel free to use and modify as needed.
