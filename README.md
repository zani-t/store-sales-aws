# TSF2 — Time Series Forecasting Pipeline

AWS-native pipeline that trains and periodically retrains store-sales forecasting models on biweekly data. Uses SARIMAX per family/store, an XGBoost stacking ensemble, and automated orchestration via Step Functions.

Built on the [Kaggle Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset. **Dataset files are not included in this repository** and must be obtained separately under Kaggle's terms.

## Architecture

S3 uploads trigger a serialized job queue that runs one biweek at a time through preprocessing, evaluation, and model retraining. See [ARCHITECTURE.md](ARCHITECTURE.md) for a concise overview.

## Repository layout

| Path | Purpose |
|------|---------|
| `tsf2/` | AWS CDK stacks (storage, compute, orchestration) |
| `containers/` | Docker images for preprocessing, training, and evaluation |
| `lambdas/` | Queue enqueue and Step Functions starter functions |
| `bootstrap/` | One-time scripts to seed historical data and initial models |
| `simulator/` | Local tool to replay daily uploads into S3 |
| `tests/` | Unit and CDK synthesis tests |

## Prerequisites

- Python 3.12
- [AWS CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/cli.html) and an AWS account
- Docker (for Lambda/ECS container builds)
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api) (for dataset download)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Configure AWS credentials and bootstrap CDK if needed:

```bash
cdk bootstrap
```

Deploy all stacks (defaults to `dev` environment):

```bash
cdk deploy --all -c env=dev
```

## Bootstrap (first run)

After deploy, run the bootstrap scripts in order to upload historical data and train initial models:

```bash
python bootstrap/1_raw.py
python bootstrap/2_sarimax_prime.py
python bootstrap/3_sarimax.py
python bootstrap/4_xgboost_prime.py
python bootstrap/5_xgboost.py
```

`1_raw.py` downloads the Kaggle competition data. Alternatively, download manually:

```bash
cd simulator && ./download.sh
```

## Simulate live ingestion

Once bootstrapped, replay daily data uploads to trigger the automated pipeline:

```bash
ENV_NAME=dev python simulator/simulator.py 14   # upload next 14 days
```

## Development

```bash
cdk synth
pytest
```

## Cost

The pipeline is event-driven: no always-on workers. Costs are near-zero when idle and incurred mainly during biweekly Fargate training/evaluation runs. Destroy non-prod stacks when not in use:

```bash
cdk destroy --all -c env=dev
```

## License

MIT — see [LICENSE](LICENSE). Kaggle dataset usage is subject to Kaggle's competition terms.
