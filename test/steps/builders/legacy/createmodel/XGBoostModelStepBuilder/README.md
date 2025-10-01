# XGBoostModelStepBuilder Test Reports

This directory contains test reports and scoring charts for the XGBoostModel step builder.

## Directory Structure

- `scoring_reports/` - Contains detailed test scoring reports and charts
  - `XGBoostModelStepBuilder_score_report.json` - Detailed test results in JSON format

## Step Information

- **Registry Name**: XGBoostModel
- **Builder Class**: XGBoostModelStepBuilder
- **Step Type**: Createmodel
- **Generated**: 2025-08-15 11:49:00

## Usage

The scoring reports are generated automatically by running:

```bash
python test/steps/builders/generate_simple_reports.py
```

Or for specific step types:

```bash
python test/steps/builders/generate_simple_reports.py --step-type createmodel
```
