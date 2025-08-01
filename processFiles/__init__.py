import azure.functions as func
import logging
import json
import os
import io
import time
import pandas as pd
import openpyxl
import numpy as np
import re
from openai import AzureOpenAI
from openai._exceptions import RateLimitError
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzedDocument
#env variables
load_dotenv()
# app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

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
                if not blob.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                    logging.warning(f"Skipping non-Excel file in traceability_matrix: {blob.name}")
                    continue

                    # Now read the workbook with openpyxl
                    try:
                        sheets = pd.read_excel(
                            io.BytesIO(file_info["content"]),
                            sheet_name=None,
                            engine="openpyxl"
                        )
                    except Exception as e:
                        logging.error(f"Failed to read Excel {blob.name}: {e}")
                        continue  # skip this file and move on

                    logging.info(f"\nTraceability Matrix '{blob.name}' contains {len(sheets)} sheets:\n")
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
    SYSTEM_INSTRUCTIONS = """
        You will receive two inputs:

        1. document_tables (all_doc_rows): A list of rows extracted from project documents. Each row contains columns such as "Requirement ID", "Description", and others.

        2. traceability_matrix (matrix_sheet): A list of rows from the requirements traceability matrix. Each row includes columns such as "Requirement ID", "Requirement Type", "Task or Activity (L4)", "Description", and more.

        Use the "Requirement ID" column from the document_tables and the "Task or Activity (L4)" column from the traceability_matrix for primary matching.

        The "Requirement ID" in document_tables corresponds to the "Task or Activity (L4)" in the traceability_matrix (ignoring any prefix).

        Example: A document row with Requirement ID RL-030-050-020 should be matched against any traceability_matrix row where "Task or Activity (L4)" or the suffix of "Requirement ID" equals RL-030-050-020, regardless of any prefix in "Requirement ID" (e.g., FR-RL-030-050-020).

        If multiple matches exist, use the "Requirement Type" column in the traceability_matrix to suggest the correct prefix (e.g., FR, BR, IP).

        For each row in document_tables:

        - If "Requirement ID" is found in "Task or Activity (L4)" in the traceability_matrix:
        - If "Description" fields differ, flag as Description Mismatch and suggest revision for clarity and consistency.
        - If both "Requirement ID" and "Description" match, suggest any further language improvements if needed.

        - If "Requirement ID" is not found, but "Description" closely matches a row in traceability_matrix, suggest the correct full "Requirement ID" (including the prefix, referencing "Requirement Type" as needed).

        - If neither "Requirement ID" nor a similar "Description" are found, flag as Missing Requirement.

        - If "Requirement ID" is missing or incomplete in the document table, suggest the appropriate prefix based on the "Requirement Type" in traceability_matrix and reconstruct the full ID as needed.

        For ambiguous cases, list all possible prefix options, referencing the "Requirement Type" column.

        Abbreviations for Requirement Type Prefixes:  
        - FR = Functional or Detailed  
        - BR = Business  
        - IP = Confirmations  

        If FR appears as the prefix, use the Requirement Type column to distinguish between Functional or Detailed types as needed.

        **Only** return the results table. **Do not** explain your process.

        **Output format**  
        Output only JSON with a top‑level comparison array of objects, each object containing exactly these keys: requirementID, description, issueType, suggestedFullRequirementID, matrixDescription, suggestedRevision, and briefExplanation. Do not output markdown tables or any other text.”. Each result object must have these properties:

        requirementID              (string)  
        description                (string)  
        issueType                  (string)  // e.g. DescriptionMismatch, SuggestCorrectID, MissingRequirement, NoIssue  
        suggestedFullRequirementID (string or null)  
        matrixDescription          (string or null)  
        suggestedRevision          (string or null)  
        briefExplanation           (string)

        """
    SYSTEM_INSTRUCTIONS2 = """
        You will receive two inputs:
        document tables(all_row_docs) and a matrix(sample_matrix_json). 
        document tables is an array of objects extracted from project documents; look for tables that have a requirementID (string) and description (string) header. 
        traceability_matrix is an array of objects from the requirements traceability matrix;
        Use the "Requirement ID" column from document tables as a key to compare the rows from document tables to the "Task Or ActivityL4" column from the matrix
        If the Requirement ID doesnt exist in the matrix, use the description column as a key and compare based on the descriptions.

        Match each document tables entry to a matrix entry by comparing requirementID in document tables to taskOrActivityL4 in the matrix. 
        For each row:
        If the ID matches but descriptions differ, issueType = "DescriptionMismatch".
        If neither ID nor similar description found, issueType = "MissingRequirement"
        suggest,if necessary, an improvement in language for the description for each requirement.
        if the issueType is neither "DescriptionMismatch" or "MissingRequirement", create a category that fits the issue or if language improvement suggested, issueType=langRevision.
        
        For the output:
        
        *output only** a JSON **array** of objects (no wrapper, no Markdown, no code‐fences, no back slashes), each object having exactly these keys: requirementID (string) description (string) issueType (string) suggestedRevision (string) briefExplanation (string) Example of the exact format you must return (on one line): [{"requirementID":"RL‑030‑010","description":"…","issueType":"MissingRequirement","suggestedRevision":"use better language","briefExplanation":"No match."}, …]
        Make absolutely sure the JSON array you return is closed with a ] and contains no extra or missing brackets.
        put Requirement ID in the Requirement ID column
        put the Issue Type in the issueType column
        put Suggestions in the Suggested Revision column
        put the explanation of the suggestions in the briefExplanation column
        """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTIONS2
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
        max_completion_tokens=10000,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    logging.info(f"Model finished with:, {response.choices[0].finish_reason}")

    agent_output = response.choices[0].message.content 
    # if agent_output.startswith('"[') and agent_output.endswith(']"'):
    #     # strip the wrapping quotes
    #     agent_output = agent_output[1:-1]
    #     # unescape interior quotes
    #     agent_output = agent_output.replace('\\"', '"')
    parsed = json.loads(agent_output)
    return parsed

def make_html_table(comparison_list):
    # 1. Define the columns in the order you want them to appear
    cols = ["requirementID", "description", "issueType", "suggestedRevision", "briefExplanation"]

    # 2. Build the header row: wrap each column name in a <th> tag
    header_cells = [f"<th>{col}</th>" for col in cols]
    header_row   = "<tr>" + "".join(header_cells) + "</tr>"

    # 3. Build each data row: for every dict in comparison_list, produce a <tr> of <td> cells
    rows_html = []
    for item in comparison_list:
        cell_html = [f"<td>{ item.get(col, '') }</td>" for col in cols]
        rows_html.append("<tr>" + "".join(cell_html) + "</tr>")
    body_rows = "\n".join(rows_html)

    # 4. Wrap it all in a scrollable <div> and a styled <table>
    html = f"""
    <div style="width:100vw; max-height:100vh; overflow:auto;">
      <table border="1" style="border-collapse:collapse; width:100%;">
        <thead style="position:sticky; top:0; background:#eee;">
          {header_row}
        </thead>
        <tbody>
          {body_rows}
        </tbody>
      </table>
    </div>
    """
    return html

def normalize_comparison_str(s: str) -> list:
    """
    Turn something like:
      "[{\"a\":1}, {\"b\":2}…]"
    or even
      "\"[{\\\"a\\\":1}, {\\\"b\\\":2}]\""
    into a real Python list of dicts.
    """
    s = s.strip()

    # 1) If it’s wrapped in extra double‐quotes, remove them
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    # 2) Unescape any embedded \" and \\n
    s = s.replace('\\"', '"').replace('\\\\\"', '"')
    s = s.replace('\\n', '').replace('\\r', '')

    # 3) Ensure it starts with [ and ends with ]
    if not s.startswith('['):
        idx = s.find('[')
        if idx != -1:
            s = s[idx:]
    if not s.endswith(']'):
        s += ']'

    # 4) Now parse
    return json.loads(s)


# @app.function_name(name="processFiles")
# @app.route(route="processFiles", auth_level=func.AuthLevel.ANONYMOUS)
def main(req: func.HttpRequest) -> func.HttpResponse:
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
        
        full_matrix_json = (
            matrix_sheet.replace({pd.NA: None, np.nan: None, float('nan'): None})
            .to_dict(orient="records")
        )    

        sample_matrix_json = full_matrix_json[:500]

        comparison = compare_files(
            foundry_url,
            foundry_key,
            all_doc_rows,
            sample_matrix_json)
        html_Table = make_html_table(comparison)

        result = {
            "summary": {
                "documentsProcessed": len(documents),
                "tablesExtracted":   len(docuTables)
            },
            "comparison": comparison,
            "htmlTable" : html_Table
        }
        return func.HttpResponse(
            body=json.dumps(result, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )

       
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(f"Internal error: {str(e)}", status_code=500)
  