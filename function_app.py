import azure.functions as func
import logging
import json
import os
import io
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzedDocument

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

def extract_tables_from_file(fcontent):

    #API Endpoint/Key/Client to connect           
    endpoint = os.getenv("document_int_endpoint")
    key = os.getenv("document_int_key")
    client = DocumentIntelligenceClient(
        endpoint= endpoint,
        credential = AzureKeyCredential(key))
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

@app.function_name(name="processFiles")
@app.route(route="processFiles", auth_level=func.AuthLevel.ANONYMOUS)
def process_Files(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info('Starting file processing')

        #Blob Storage Access
        load_dotenv()
        sas_url = os.getenv("blob_sas_url")
        sas_token = os.getenv("blob_sas_token")
        container = os.getenv("container")
        folders = ["documents/","traceability_matrix/"]

        if not sas_url or not sas_token or not container:
            raise ValueError("Missing env vars")

        blob_service_client = BlobServiceClient(account_url = sas_url, credential= sas_token)
        container_client = blob_service_client.get_container_client(container)

        documents = []
        traceabilty_matrix = []

        #loop through each folder and append documents in blob to respective list 
        for folder in folders:
            print(f"\nProcessing: {folder}")

            #grab files that start with the names in folders list 
            blobs = container_client.list_blobs(name_starts_with = folder)
            #loop through each file
            for blob in blobs:
                #SKIP empty.txt placeholder file
                if blob.name.endswith("empty.txt"):
                    continue
                print(f"    Found File: {blob.name}")
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
                    #print statement to make sure we're pulling the right data
                    print(f"\nTracebility Matrix '{blob.name}' contains {len(sheets)} sheets:\n")

                    #sheet_n = keys of sheets dictionary, df= the data frame value for the respective sheet
                    for sheet_n, df in sheets.items():
                        #print the dimensions of each sheet | .shape attribute used with panda dataframes to quickly see dimentions | .shape[0] = rows | .shape[1] = columns
                        print(f"    Sheet '{sheet_n}': {df.shape[0]} rows x {df.shape[1]} columns")
                    traceabilty_matrix.append({
                        "name": blob.name,
                        "content": file_info["content"],
                        "sheets": sheets
                    })

        docuTables = []
        for doc in documents:
            #extract tables using extract_tables_from_file
            print(f"\nSending {doc["name"]} to Document Intelligence to extract... ")
            dfs = extract_tables_from_file(doc["content"])
            #append tables into docuTables list
            docuTables.extend(dfs)

        summary = {
            "documentsProcessed":len(documents),
            "tablesExtracted":len(docuTables),
            "matrixSheets":list(traceabilty_matrix[0]["sheets"].keys()) if traceabilty_matrix else []
        }

        return func.HttpResponse(
            json.dumps(summary),
            status_code=200,
            mimetype="application/json"
        )

       
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Internal error: {str(e)}", status_code=500)
  