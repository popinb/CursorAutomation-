# Alternative Cell 1 - Careful dependency management
# This version installs packages in a specific order to minimize conflicts

# COMMAND ----------

# Install packages in careful order to minimize conflicts
# Don't uninstall anything - just upgrade what we need

# Step 1: Upgrade MLflow first (most important for visualizations)
!pip install --upgrade mlflow>=3.0 --quiet --no-deps

# Step 2: Install MLflow dependencies
!pip install mlflow[genai]>=3.0 --quiet

# Step 3: Install LangChain (needed for MLflow metrics)
!pip install langchain-core --quiet
!pip install langchain-community --quiet  
!pip install langchain-openai --quiet

# Step 4: Other packages
!pip install pandas plotly python-docx --quiet

# Note: If you still get protobuf errors, they're just warnings
# The notebook will still work - Databricks handles the compatibility

# Restart Python kernel
dbutils.library.restartPython()

# COMMAND ----------