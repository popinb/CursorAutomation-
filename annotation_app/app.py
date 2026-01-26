"""
Text Annotation App
A modern, user-friendly annotation tool built with Streamlit.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# App configuration
st.set_page_config(
    page_title="Annotation App",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card-like containers */
    .annotation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Text to annotate styling */
    .text-display {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    /* Progress indicator */
    .progress-text {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    /* Label buttons */
    .stButton > button {
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Navigation buttons */
    .nav-button {
        width: 100%;
    }
    
    /* Header styling */
    h1 {
        color: #333;
        font-weight: 600;
    }
    
    /* Success message */
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize session state
def init_session_state():
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'labels' not in st.session_state:
        st.session_state.labels = ["Positive", "Negative", "Neutral"]
    if 'project_name' not in st.session_state:
        st.session_state.project_name = "default_project"
    if 'show_success' not in st.session_state:
        st.session_state.show_success = False

init_session_state()

# Helper functions
def load_data_from_file(uploaded_file):
    """Load data from uploaded file (CSV or JSON)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Assume first column contains text to annotate
            text_column = df.columns[0]
            return df[text_column].tolist()
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    return data
                elif all(isinstance(item, dict) for item in data):
                    # Try common keys
                    for key in ['text', 'content', 'sentence', 'data']:
                        if key in data[0]:
                            return [item[key] for item in data]
            return []
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def save_annotations():
    """Save current annotations to file."""
    output = {
        "project": st.session_state.project_name,
        "labels": st.session_state.labels,
        "timestamp": datetime.now().isoformat(),
        "annotations": []
    }
    
    for idx, text in enumerate(st.session_state.data):
        annotation_entry = {
            "id": idx,
            "text": text,
            "label": st.session_state.annotations.get(idx, None),
            "annotated": idx in st.session_state.annotations
        }
        output["annotations"].append(annotation_entry)
    
    filename = DATA_DIR / f"{st.session_state.project_name}_annotations.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    return filename

def load_existing_annotations(project_name):
    """Load existing annotations for a project."""
    filename = DATA_DIR / f"{project_name}_annotations.json"
    if filename.exists():
        with open(filename, 'r') as f:
            data = json.load(f)
            st.session_state.labels = data.get('labels', st.session_state.labels)
            st.session_state.data = [a['text'] for a in data['annotations']]
            st.session_state.annotations = {
                a['id']: a['label'] 
                for a in data['annotations'] 
                if a['annotated']
            }
            return True
    return False

def get_stats():
    """Calculate annotation statistics."""
    total = len(st.session_state.data)
    annotated = len(st.session_state.annotations)
    remaining = total - annotated
    
    label_counts = {}
    for label in st.session_state.annotations.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    return {
        "total": total,
        "annotated": annotated,
        "remaining": remaining,
        "progress": (annotated / total * 100) if total > 0 else 0,
        "label_counts": label_counts
    }

def annotate(label):
    """Apply annotation to current item."""
    st.session_state.annotations[st.session_state.current_index] = label
    st.session_state.show_success = True
    # Auto-advance to next unannotated item
    if st.session_state.current_index < len(st.session_state.data) - 1:
        st.session_state.current_index += 1

def navigate(direction):
    """Navigate between items."""
    if direction == "next" and st.session_state.current_index < len(st.session_state.data) - 1:
        st.session_state.current_index += 1
    elif direction == "prev" and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
    st.session_state.show_success = False

def jump_to_next_unannotated():
    """Jump to the next unannotated item."""
    for i in range(len(st.session_state.data)):
        if i not in st.session_state.annotations:
            st.session_state.current_index = i
            break

# Main UI
def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏷️ Annotation App")
        st.caption("A modern tool for text annotation and labeling")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Project name
        st.session_state.project_name = st.text_input(
            "Project Name", 
            value=st.session_state.project_name
        )
        
        st.divider()
        
        # Data upload section
        st.subheader("📂 Data")
        
        # Load existing project
        existing_projects = list(DATA_DIR.glob("*_annotations.json"))
        if existing_projects:
            project_names = [p.stem.replace("_annotations", "") for p in existing_projects]
            selected_project = st.selectbox(
                "Load Existing Project",
                ["-- Select --"] + project_names
            )
            if selected_project != "-- Select --":
                if st.button("Load Project"):
                    if load_existing_annotations(selected_project):
                        st.session_state.project_name = selected_project
                        st.session_state.current_index = 0
                        st.success(f"Loaded project: {selected_project}")
                        st.rerun()
        
        # Upload new data
        uploaded_file = st.file_uploader(
            "Upload Data (CSV or JSON)",
            type=['csv', 'json']
        )
        
        if uploaded_file:
            if st.button("Load Data"):
                data = load_data_from_file(uploaded_file)
                if data:
                    st.session_state.data = data
                    st.session_state.annotations = {}
                    st.session_state.current_index = 0
                    st.success(f"Loaded {len(data)} items!")
                    st.rerun()
        
        # Or use sample data
        if st.button("Load Sample Data"):
            st.session_state.data = [
                "The product quality exceeded my expectations. Highly recommend!",
                "Terrible customer service. Never buying again.",
                "It's okay, nothing special but does the job.",
                "Absolutely love this! Best purchase I've made this year.",
                "The delivery was late and the packaging was damaged.",
                "Great value for money. Would buy again.",
                "Not what I expected based on the description.",
                "Average product, average price. Fair deal.",
                "Customer support was very helpful in resolving my issue.",
                "The instructions were confusing and incomplete."
            ]
            st.session_state.annotations = {}
            st.session_state.current_index = 0
            st.success("Loaded 10 sample items!")
            st.rerun()
        
        st.divider()
        
        # Labels configuration
        st.subheader("🏷️ Labels")
        
        # Display current labels
        st.write("Current labels:")
        for i, label in enumerate(st.session_state.labels):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {label}")
            with col2:
                if st.button("×", key=f"del_{i}", help=f"Remove {label}"):
                    st.session_state.labels.pop(i)
                    st.rerun()
        
        # Add new label
        new_label = st.text_input("Add new label")
        if st.button("Add Label") and new_label:
            if new_label not in st.session_state.labels:
                st.session_state.labels.append(new_label)
                st.rerun()
        
        st.divider()
        
        # Export section
        st.subheader("💾 Export")
        if st.button("Save Annotations", type="primary"):
            if st.session_state.data:
                filename = save_annotations()
                st.success(f"Saved to {filename}")
        
        # Download button
        if st.session_state.annotations:
            stats = get_stats()
            export_data = {
                "project": st.session_state.project_name,
                "total_items": stats["total"],
                "annotated_items": stats["annotated"],
                "labels": st.session_state.labels,
                "annotations": [
                    {
                        "text": st.session_state.data[idx],
                        "label": label
                    }
                    for idx, label in st.session_state.annotations.items()
                ]
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"{st.session_state.project_name}_export.json",
                mime="application/json"
            )
    
    # Main content area
    if not st.session_state.data:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>Welcome to the Annotation App! 👋</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Get started by uploading your data or loading sample data from the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start guide
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### How to use this app:
            
            1. **Upload Data**: Use the sidebar to upload a CSV or JSON file containing text to annotate
            2. **Configure Labels**: Add or remove labels that fit your annotation task
            3. **Annotate**: Click on labels to annotate each text item
            4. **Navigate**: Use Previous/Next buttons or keyboard shortcuts
            5. **Export**: Save your annotations as JSON for later use
            
            ### Supported file formats:
            - **CSV**: First column should contain the text to annotate
            - **JSON**: Array of strings or objects with 'text' field
            """)
    else:
        # Statistics row
        stats = get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", stats["total"])
        with col2:
            st.metric("Annotated", stats["annotated"])
        with col3:
            st.metric("Remaining", stats["remaining"])
        with col4:
            st.metric("Progress", f"{stats['progress']:.1f}%")
        
        # Progress bar
        st.progress(stats["progress"] / 100)
        
        st.divider()
        
        # Current item display
        current_idx = st.session_state.current_index
        current_text = st.session_state.data[current_idx]
        current_annotation = st.session_state.annotations.get(current_idx)
        
        # Item header
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.subheader(f"Item {current_idx + 1} of {len(st.session_state.data)}")
        with col2:
            if current_annotation:
                st.success(f"✓ Labeled: **{current_annotation}**")
        with col3:
            if st.button("🎯 Next Unlabeled"):
                jump_to_next_unannotated()
                st.rerun()
        
        # Text display
        st.markdown(f"""
        <div class="text-display">
            {current_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Label buttons
        st.write("**Select a label:**")
        
        # Create columns for label buttons
        cols = st.columns(len(st.session_state.labels))
        for i, label in enumerate(st.session_state.labels):
            with cols[i]:
                button_type = "primary" if current_annotation == label else "secondary"
                if st.button(
                    f"{'✓ ' if current_annotation == label else ''}{label}",
                    key=f"label_{label}",
                    type=button_type,
                    use_container_width=True
                ):
                    annotate(label)
                    st.rerun()
        
        # Navigation
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Previous", disabled=current_idx == 0, use_container_width=True):
                navigate("prev")
                st.rerun()
        
        with col2:
            # Jump to specific item
            new_idx = st.number_input(
                "Go to item:",
                min_value=1,
                max_value=len(st.session_state.data),
                value=current_idx + 1,
                label_visibility="collapsed"
            )
            if new_idx - 1 != current_idx:
                st.session_state.current_index = new_idx - 1
                st.rerun()
        
        with col3:
            if st.button(
                "Next →", 
                disabled=current_idx == len(st.session_state.data) - 1,
                use_container_width=True
            ):
                navigate("next")
                st.rerun()
        
        # Label distribution chart
        if stats["label_counts"]:
            st.divider()
            st.subheader("📊 Label Distribution")
            
            chart_data = pd.DataFrame(
                list(stats["label_counts"].items()),
                columns=["Label", "Count"]
            )
            st.bar_chart(chart_data.set_index("Label"))
        
        # Annotations table (collapsible)
        with st.expander("📋 View All Annotations"):
            if st.session_state.annotations:
                table_data = []
                for idx in sorted(st.session_state.annotations.keys()):
                    table_data.append({
                        "ID": idx + 1,
                        "Text": st.session_state.data[idx][:100] + "..." 
                                if len(st.session_state.data[idx]) > 100 
                                else st.session_state.data[idx],
                        "Label": st.session_state.annotations[idx]
                    })
                st.dataframe(
                    pd.DataFrame(table_data),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No annotations yet. Start labeling!")

if __name__ == "__main__":
    main()
