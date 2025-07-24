import azure.functions as func
import logging
import json
import os
import io
import time
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from openai.error import RateLimitError
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzedDocument
#env variables
load_dotenv()
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

def grab_from_blob(sas_url,sas_token,container):
    logging.info('Grabbing files...')

    #Blob Storage Access
    
    folders = ["documents/","traceability_matrix/"]

    if not sas_url or not sas_token or not container:
        raise ValueError("Missing env vars")

    blob_service_client = BlobServiceClient(account_url = sas_url, credential= sas_token)
    container_client = blob_service_client.get_container_client(container)

    documents = []
    traceabilty_matrix = []

    #loop through each folder and append documents in blob to respective list 
    for folder in folders:
        logging.info(f"\nProcessing: {folder}")

        #grab files that start with the names in folders list 
        blobs = container_client.list_blobs(name_starts_with = folder)
        #loop through each file
        for blob in blobs:
            #SKIP empty.txt placeholder file
            if blob.name.endswith("empty.txt"):
                continue
            logging.info(f"    Found File: {blob.name}")
            #dowload file contents as binary and store in memory
            blob_client = container_client.get_blob_client(blob)
            file_data = blob_client.download_blob().readall()

            #store files as dictionary, {name of file, content of file}
            file_info = {
                "name": blob.name,
                "content":file_data
            }

            #append files to respective folder list
            if folder == "documents/":
                documents.append(file_info)

            #WHEW the traceability matrix has multiple sheets so we gotta make sure we can read each one
            elif folder == "traceability_matrix/":
                #io.bytesIO wraps binary from file_data as a file like object so pandas can read it/sheet_name = none tells pandas to load all sheets in the excel workbook
                #create a dictionary where each key is a sheet name and its value is a dataframe of that sheet
                sheets = pd.read_excel(io.BytesIO(file_info["content"]), sheet_name=None)
                #statement to make sure we're pulling the right data
                logging.info(f"\nTracebility Matrix '{blob.name}' contains {len(sheets)} sheets:\n")

                #sheet_n = keys of sheets dictionary, df= the data frame value for the respective sheet
                for sheet_n, df in sheets.items():
                    #print the dimensions of each sheet | .shape attribute used with panda dataframes to quickly see dimentions | .shape[0] = rows | .shape[1] = columns
                    logging.info(f"    Sheet '{sheet_n}': {df.shape[0]} rows x {df.shape[1]} columns")
                traceabilty_matrix.append({
                    "name": blob.name,
                    "content": file_info["content"],
                    "sheets": sheets
                })
    return documents, traceabilty_matrix

def extract_tables_from_file(docint_endpoint,docint_key,fcontent):
    #API Endpoint/Key/Client to connect           
    
    client = DocumentIntelligenceClient(
        endpoint= docint_endpoint,
        credential = AzureKeyCredential(docint_key))
    result = None

    with io.BytesIO(fcontent) as f: 
        docu = f.read()
    #call api and send over the document using layout preset
    poller = client.begin_analyze_document(model_id="prebuilt-layout", body = docu)
    #wait for results from api
    result = poller.result()

    #data frame tables from document
    table_data_frames = []
    #loop through only tables in result to create 2d matrix of row and columns with 2 variables,an index of each table, and the table itself
    for i, table in enumerate(result.tables):
        max_row = max(cell.row_index for cell in table.cells) + 1
        max_col = max(cell.column_index for cell in table.cells) + 1
        table_matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
        #loop through cells and add content to table_matrix 
        for cell in table.cells:
            table_matrix[cell.row_index][cell.column_index] = cell.content
        #add each row list item in table_matrix and append to pandas data frame
        df = pd.DataFrame(table_matrix)
        table_data_frames.append(df)
    return table_data_frames

def safe_chat_completion(client, **params):
    """
    Wraps client.chat.completions.create(...) with retries on 429.
    """
    max_retries = 5
    backoff = 1  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**params)
        except RateLimitError as e:
            # Azure OpenAI sometimes includes a Retry-After header
            retry_after = None
            if hasattr(e, "headers") and e.headers.get("Retry-After"):
                retry_after = int(e.headers["Retry-After"])
            else:
                retry_after = backoff

            if attempt == max_retries:
                raise  # no retries left

            print(f"[429] Rate limited. Retry #{attempt} in {retry_after}s…")
            time.sleep(retry_after)
            backoff *= 2  # exponential backoff

    # Should never get here, since we either return or raise
    raise RuntimeError("Exceeded retries for rate limit")

def compare_files(foundry_url,foundry_key,all_doc_rows,matrix_json):
    endpoint = foundry_url
    model_name = "gpt-4.1"
    deployment = "compareFiles"

    subscription_key = foundry_key
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert project‑documentation assistant. "
                "Compare document_tables against traceability_matrix and flag issues "
                "per the predefined rules."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "document_tables": all_doc_rows,
                "traceability_matrix": matrix_json
            }, ensure_ascii=False)
        }
    ]

    response = safe_chat_completion(
        client,
        messages=messages,
        model="compareFiles",
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].message.content

@app.function_name(name="processFiles")
@app.route(route="processFiles", auth_level=func.AuthLevel.ANONYMOUS)
def processFiles(req: func.HttpRequest) -> func.HttpResponse:
    try:
        sas_url = os.getenv("blob_sas_url")
        sas_token = os.getenv("blob_sas_token")
        container = os.getenv("container")

        docint_endpoint = os.getenv("document_int_endpoint")
        docint_key = os.getenv("document_int_key")

        foundry_url=os.getenv("foundry_url")
        foundry_key=os.getenv("foundry_key")

        for name, val in [
            ("blob_sas_url", sas_url), ("blob_sas_token", sas_token),
            ("container", container), ("document_int_endpoint", docint_endpoint),
            ("document_int_key", docint_key), ("foundry_url", foundry_url),
            ("foundry_key", foundry_key),
        ]:
            if not val:
                raise ValueError(f"Missing env var: {name}")
            
        #grab files from blob storage
        documents, matrix = grab_from_blob(
            sas_url,
            sas_token,
            container)
        #loop through each document and grab only the tables
        docuTables = []
        for doc in documents:
            #extract tables using document intelligence function
            logging.info(f"\nSending {doc['name']} to Document Intelligence to extract…")
            table_dfs = extract_tables_from_file(docint_endpoint,docint_key,doc['content'])
            #append tables into docuTables list
            docuTables.extend(table_dfs)

        matrix_sheet = matrix[0]["sheets"]["Requirements 1.23"]    
        #Turn data frames into json list
        all_doc_rows = []
        for df in docuTables:
            clean_df = df.replace({pd.NA: None, np.nan: None, float('nan'): None})
            all_doc_rows.extend(clean_df.to_dict(orient="records"))
        matrix_json = (
            matrix_sheet
            .replace({pd.NA: None, np.nan: None, float('nan'): None})
            .to_dict(orient="records")
        )    

        comparison = compare_files(
            foundry_url,
            foundry_key,
            all_doc_rows,
            matrix_json)

        result = {
            "summary": {
                "documentsProcessed": len(documents),
                "tablesExtracted": len(docuTables)
            },
            "comparison": comparison
        }

        return func.HttpResponse(
            body=json.dumps(result, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

       
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Internal error: {str(e)}", status_code=500)
  