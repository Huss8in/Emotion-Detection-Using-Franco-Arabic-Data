import os
from dotenv import load_dotenv
import pymssql
import pymongo
import pandas as pd
from sqlalchemy import create_engine


# Load environment variables from .env file
load_dotenv()
#--------------------------Mongo_Connection--------------------------#
# MongoDB connection string
mongo_connection_string = os.getenv("botit_sample_connection_string") # botit_dev_connection_string || botit_sample_connection_string
mongo_client = pymongo.MongoClient(mongo_connection_string)
mongo_db = mongo_client['botit-sample'] # botitdev || botit-sample

#--------------------------Change-Stream--------------------------#
# # Change Streams for all collections
# change_streams = [
#     mongo_db['Orders'].watch(full_document='updateLookup'),
#     mongo_db['Vendors'].watch(full_document='updateLookup'),
#     mongo_db['Items'].watch(full_document='updateLookup'),
#     mongo_db['Status'].watch(full_document='updateLookup')
# ]
# # Process Change Events for all collections
# for change_stream in change_streams:
#     for change in change_stream:
#         print(change)
#--------------------------aggregation_pipelines--------------------------#
# Retrieve data from the MongoDB collections using the pipelines
# Perform the aggregation piplines to filter needed column only

orders_pipeline = [{"$project": {"_id": 1, "_vendor": 1, "branch": 1, "deliveryDay": 1, "paymentMethod": 1, "price": 1, "createdAt": 1, "status": 1}}]
df_orders = pd.DataFrame(list(mongo_db['Orders'].aggregate(orders_pipeline)))

vendors_pipeline = [{"$project": {"_id": 1, "name": 1, "status": 1, "integration": 1, "shoppingCategory": 1}}]
df_vendors = pd.DataFrame(list(mongo_db['Vendors'].aggregate(vendors_pipeline)))  

items_pipeline = [{"$project": {"_id": 1, "_vendor": 1, "name": 1, "data": 1, "variants": 1, "price": 1, "status": 1}}]
df_Items = pd.DataFrame(list(mongo_db['Items'].aggregate(items_pipeline)))

# batch_size = 1000

# # Extract the first 100 rows from each collection using limit
# df_orders = pd.DataFrame(list(mongo_db['Orders'].aggregate([{"$project": {"_id": 1, "_vendor": 1, "branch": 1, "deliveryDay": 1, "paymentMethod": 1, "price": 1, "createdAt": 1, "status": 1}}, {"$limit": batch_size}])))
# df_vendors = pd.DataFrame(list(mongo_db['Vendors'].aggregate([{"$project": {"_id": 1, "name": 1, "status": 1, "integration": 1, "shoppingCategory": 1}}, {"$limit": batch_size}])))
# df_Items = pd.DataFrame(list(mongo_db['Items'].aggregate([{"$project": {"_id": 1, "_vendor": 1, "name": 1, "data": 1, "variants": 1, "price": 1, "status": 1}}, {"$limit": batch_size}])))

# Transformation for ** Orders ** DataFrame
df_orders['price_subtotal'] = df_orders['price'].apply(lambda x: x.get('subtotal') if isinstance(x, dict) else None)
df_orders['price_delivery'] = df_orders['price'].apply(lambda x: x.get('delivery') if isinstance(x, dict) else None)
df_orders['price_total'] = df_orders['price'].apply(lambda x: x.get('total') if isinstance(x, dict) else None)
df_orders['price_discount'] = df_orders['price'].apply(lambda x: x.get('discount') if isinstance(x, dict) else None)
df_orders['price_wallet'] = df_orders['price'].apply(lambda x: x.get('wallet') if isinstance(x, dict) else None)

# Transformation for ** Vendors ** DataFrame
df_vendors['name'] = df_vendors['name'].apply(lambda x: x.get('en') if isinstance(x, dict) and 'en' in x else x)
df_vendors['integration'] = df_vendors['integration'].apply(lambda x: x.get('system') if isinstance(x, dict) else None)
df_vendors['status'] = df_vendors['status'].replace('nan', None)
df_vendors = df_vendors[['_id', 'name', 'status', 'integration', 'shoppingCategory']]

# Transformation for ** Items ** DataFrame
df_Items['name_en'] = df_Items['name'].apply(lambda x: x.get('en') if isinstance(x, dict) and 'en' in x else None)
df_Items['shoppingCategory_en'] = df_Items['data'].apply(lambda x: x['shoppingCategory'].get('en') if isinstance(x, dict) and 'shoppingCategory' in x and isinstance(x['shoppingCategory'], dict) else None)
df_Items['shoppingSubcategory_en'] = df_Items['data'].apply(lambda x: x['shoppingSubcategory'].get('en') if isinstance(x, dict) and 'shoppingSubcategory' in x and isinstance(x['shoppingSubcategory'], dict) else None)
df_Items['itemCategory_en'] = df_Items['data'].apply(lambda x: x['itemCategory'].get('en') if isinstance(x, dict) and 'itemCategory' in x and isinstance(x['itemCategory'], dict) else None)
df_Items['variants_updated_at'] = df_Items['variants'].apply(lambda x: x[0]['updated_at'] if isinstance(x, list) and len(x) > 0 and 'updated_at' in x[0] else None)
df_Items['variants_created_at'] = df_Items['variants'].apply(lambda x: x[0]['created_at'] if isinstance(x, list) and len(x) > 0 and 'created_at' in x[0] else None)
df_Items['variants_updated_at'] = df_Items['variants_updated_at'].replace(pd.NaT, None)
df_Items['variants_created_at'] = df_Items['variants_created_at'].replace(pd.NaT, None)

df_Items = df_Items[['_id', '_vendor', 'name_en', 'shoppingCategory_en', 'shoppingSubcategory_en', 'itemCategory_en', 'price', 'status', 'variants_updated_at', 'variants_created_at']]

# Now proceed with your batch insertion logic for the df_Items DataFrame


# status_data
status_data = []
df_status = pd.DataFrame(columns=['order_id', 'status_id', 'status_name', 'status_source', 'status_created_at','new_status_created_at', 'status_updated_at'])

if 'status' in df_orders.columns:
    for _, row in df_orders.iterrows():
        order_id = row['_id']
        if isinstance(row['status'], list):
            for status_entry in row['status']:
                if isinstance(status_entry, dict):
                    status_data.append([
                        order_id,
                        status_entry.get('_id'),
                        status_entry.get('name'),
                        status_entry.get('source'),
                        status_entry.get('createdAt'),
                        status_entry.get('created_at'), # +++--------new--------+++
                        status_entry.get('updatedAt')
                    ])
                else:
                    status_data.append([order_id, status_entry, None, None, None, None, None])

    df_status = pd.DataFrame(status_data, columns=['order_id', 'status_id', 'status_name', 'status_source', 'status_created_at','new_status_created_at', 'status_updated_at'])
    
df_orders = df_orders[['_id', '_vendor', 'deliveryDay', 'paymentMethod', 'price_subtotal','price_delivery', 'price_total', 'price_discount', 'price_wallet',  'createdAt']]
#--------------------------SQL_Connection--------------------------#

# SQL Server connection details
sql_server_host = 'localhost'
sql_server_user = 'SA'
sql_server_password = 'DockerImage}'
sql_server_database = ''


try:
    # Connect to SQL Server
    sql_conn = pymssql.connect(server=sql_server_host, user=sql_server_user, password=sql_server_password, database=sql_server_database)
    print(f"Connected to SQL Server at {sql_server_host} successfully!")

    # Define the connection string for SQLAlchemy
    conn_str = f'mssql+pymssql://{sql_server_user}:{sql_server_password}@{sql_server_host}/{sql_server_database}'

    # Create SQLAlchemy engine
    engine = create_engine(conn_str)

    #--------------------------Data_Processing--------------------------#

    # Truncate strings in DataFrames to avoid exceeding column lengths in SQL Server
    max_length_orders = df_orders.astype(str).applymap(len).max().max()
    max_length_vendors = df_vendors.astype(str).applymap(len).max().max()
    max_length_items = df_Items.astype(str).applymap(len).max().max()
    max_length_status = df_status.astype(str).applymap(len).max().max()

    df_orders = df_orders.astype(str).apply(lambda x: x.str[:max_length_orders])
    df_vendors = df_vendors.astype(str).apply(lambda x: x.str[:max_length_vendors])
    df_Items = df_Items.astype(str).apply(lambda x: x.str[:max_length_items])
    df_status = df_status.astype(str).apply(lambda x: x.str[:max_length_status])

    #--------------------------Data_Insertion--------------------------#

    batch_size = 1000  # Define your batch size

    # Insert DataFrames into SQL Server tables in batches
    for start in range(0, len(df_orders), batch_size):
        end = start + batch_size
        df_orders[start:end].to_sql(name='Orders', con=engine, if_exists='append', index=False, method='multi')

    for start in range(0, len(df_vendors), batch_size):
        end = start + batch_size
        df_vendors[start:end].to_sql(name='Vendors', con=engine, if_exists='append', index=False, method='multi')

    for start in range(0, len(df_Items), batch_size):
        end = start + batch_size
        df_Items[start:end].to_sql(name='Items', con=engine, if_exists='append', index=False, method='multi')

    for start in range(0, len(df_status), batch_size):
        end = start + batch_size
        df_status[start:end].to_sql(name='Status', con=engine, if_exists='append', index=False, method='multi')

    print("Data inserted successfully!")

except Exception as e:
    print("An error occurred:", e)
finally:
    # Close the connections
    if 'sql_conn' in locals():
        sql_conn.close()
    engine.dispose()
    mongo_client.close()