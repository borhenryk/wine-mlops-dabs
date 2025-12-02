# Wine MLOps DABs Project 🍷

End-to-end ML pipeline for wine quality classification using **Databricks Asset Bundles (DABs)**.

## 🏗️ Project Structure

```
wine-mlops-dabs/
├── databricks.yml              # Main bundle configuration
├── resources/
│   └── wine_pipeline_job.yml   # Job definitions
├── src/wine_mlops/
│   ├── 01_load_data.py         # Download & save wine dataset
│   ├── 02_train_model.py       # Train RandomForest classifier
│   ├── 03_validate_model.py    # Validate model quality
│   └── 04_deploy_model.py      # Register to Unity Catalog
└── .github/workflows/
    └── ci.yml                  # CI/CD pipeline
```

## 🚀 Deployment Targets

| Target | Catalog | Schema | Trigger |
|--------|---------|--------|--------|
| `dev` | mcp_dabs_test | wine_mlops_dev | Manual / develop branch |
| `staging` | mcp_dabs_test | wine_mlops_staging | main branch |
| `prod` | mcp_dabs_test | wine_mlops_prod | After staging (main) |

## 🔧 Setup

### 1. GitHub Secrets

Add these secrets to your repository:

| Secret | Value |
|--------|-------|
| `DATABRICKS_HOST` | `https://dbc-cc0aa83c-12fd.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | Your Databricks PAT |

### 2. Local Development

```bash
# Install Databricks CLI
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Configure authentication
export DATABRICKS_HOST=https://dbc-cc0aa83c-12fd.cloud.databricks.com
export DATABRICKS_TOKEN=<your-token>

# Validate bundle
databricks bundle validate -t dev

# Deploy to dev
databricks bundle deploy -t dev

# Run the pipeline
databricks bundle run -t dev wine_training_pipeline
```

## 📊 Pipeline Overview

1. **Load Data**: Downloads UCI Wine dataset, saves to Delta table
2. **Train Model**: RandomForest classifier with MLflow tracking
3. **Validate Model**: Checks accuracy >= 85%, F1 >= 80%
4. **Deploy Model**: Registers to Unity Catalog with Champion alias

## 🔄 CI/CD Flow

```
push to develop → Validate → Deploy to Dev
push to main    → Validate → Deploy to Staging → Deploy to Prod
manual trigger  → Validate → Deploy to selected target + Run pipeline
```

## 📝 License

MIT
