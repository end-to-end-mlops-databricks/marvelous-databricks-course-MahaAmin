import yaml
import logging

from fraud_credit_cards.data_processor import DataProcessor
from fraud_credit_cards.fraud_model import FraudModel

# load configurations
with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Configuration loaded: ")
print(yaml.dump(config, default_flow_style=False))
