# Alternative Cell 1 - Minimal Installation
# This version installs only what's absolutely necessary

# COMMAND ----------

# Install required packages - minimal version to avoid conflicts
# This approach installs only the essential packages

# Core packages for the notebook
!pip install mlflow>=3.0 --quiet
!pip install openai --quiet  # Direct OpenAI instead of langchain
!pip install pandas --quiet
!pip install plotly --quiet
!pip install python-docx --quiet

# Note: If you still get conflicts, you can use this ultra-minimal version:
# !pip install mlflow pandas openai --quiet --no-deps

# Restart Python kernel
dbutils.library.restartPython()

# COMMAND ----------