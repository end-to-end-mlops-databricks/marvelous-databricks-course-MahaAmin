# Commands

### Install Databricks CLI

```
brew tap databricks/tap

brew install databricks
```

### Initiate Authenticaton

```
databricks auth login --configure-cluster --host <workspace-url>
```

### Show Databricks Profiles

```
databricks auth profiles
```

### Copy Data to Databricks Volumes

```
databricks fs cp data/data.csv dbfs:/Volumes/mlops_dev/house_prices/data/data.csv
```