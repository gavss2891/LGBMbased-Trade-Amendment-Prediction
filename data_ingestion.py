# Celonis connection and trade data ingestion.
# Credentials are loaded from a .env file in the working directory.
# Required variables: CELONIS_URL, CELONIS_API_TOKEN, CELONIS_POOL_ID,
#                     CELONIS_SPACE_ID, CELONIS_PACKAGE_ID, CELONIS_KNOWLEDGE_MODEL_ID

from pycelonis import get_celonis
import pycelonis.pql as pql
from pycelonis.pql import PQL, PQLColumn, PQLFilter
from pycelonis.pql.saola_connector import KnowledgeModelSaolaConnector
import pandas as pd
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os

load_dotenv()

url               = os.getenv('CELONIS_URL')
api_token         = os.getenv('CELONIS_API_TOKEN')
pool_id           = os.getenv('CELONIS_POOL_ID')
space_id          = os.getenv('CELONIS_SPACE_ID')
package_id        = os.getenv('CELONIS_PACKAGE_ID')
knowledge_model_id = os.getenv('CELONIS_KNOWLEDGE_MODEL_ID')

login = {
    'base_url':  url,
    'api_token': api_token,
}

celonis          = get_celonis(**login)
data_pool        = celonis.data_integration.get_data_pool(pool_id)
space            = celonis.studio.get_space(space_id)
package          = space.get_package(package_id)
knowledge_model  = package.get_knowledge_model(knowledge_model_id)
content          = knowledge_model.get_content()
data_model_id    = content.data_model_id

# Inspect available attributes in the BASE_TRADE_FOR_ML record before
# modifying the attribute list below.
records    = content.records
record     = records.find_by_id('BASE_TRADE_FOR_ML')
attributes = record.attributes

print("Attribute ID | Display Name | Column Type")
print("-" * 80)
for attr in attributes:
    print(f"{attr.id:30s} | {attr.display_name:30s} | {attr.column_type}")

# Attributes pulled via PQL. Identifiers and metadata-only columns are retained
# for downstream analysis but excluded from model features at the split stage.
needed_attribute_ids = [
    'TRADEENTRYDATE',           # time of trade entry
    'ISCOUNTERPARTYCHANGED',    # target variable
    'AMENDEDONDAY',             # excluded from forecasting — for signal examination
    'NBINTERNAL',               # excluded from forecasting
    'NBEXTERNAL',               # excluded from forecasting
    'CHANGEDTOCOUNTERPARTY',    # excluded from forecasting
    'PORTFOLIO',
    'TRADETYPE',
    'INSTRUMENT',
    'VERSION',
    'PERMISSION',
    'MDSET',
    'TRPRODUCTTYPE',
    'CMTRADETYPE',
    'CLEARERDISPLAY',
    'ISSUER',
    'OMSSYMBOL',
    'BOOK',
    'TRADEFAMILY',
    'TRADEGROUP',
    'TRADETYPECODE',
    'CATEGORY',
    'TYPOLOGY',
    'FUND',
    'STRATEGY',
    'BOOKEDBY',
    'BROKER1',
    'CLEARINGHOUSE',
    'MKTSECTOR',
    'BUYSELLFLAG',
    'TRADECATEGORY',
    'RISKSUBCATEGORY',
    'HEDGINGCATEGORY',
    'FIRSTBOOKEDBY',
    'ISINTERNAL',
    'EXOTICHEDGE',
    'INSTRUMENTSYMBOL',
    'EXECUTIONVENUE',
    'COUNTERPARTY',
    'COUNTERPARTYRISK',
    'LISTEDOTC',
    'MARKET',
    'INSTRUMENTCCY',
    'BUSINESS',
]

query    = PQL()
attr_dict  = {attr.id: attr for attr in attributes}
found_attrs = []

for attr_id in needed_attribute_ids:
    if attr_id in attr_dict:
        attr = attr_dict[attr_id]
        query += PQLColumn(name=attr.display_name, query=attr.pql)
        found_attrs.append(attr_id)
    else:
        print(f"{attr_id} not found")

data_model = data_pool.get_data_model(data_model_id)

df_pql = pql.DataFrame.from_pql(
    query,
    saola_connector=KnowledgeModelSaolaConnector(data_model, knowledge_model)
)

df_global = df_pql.to_pandas()
print(f"\nShape: {df_global.shape}")
df_global.head()
