# Azure Document Comparison Function App

This repository contains an Azure Function App that automates the process of extracting tables from project documents, comparing them against a requirements traceability matrix, and returning structured results for downstream consumption (e.g., Power Apps).

---

## 🚀 Overview

1. **grab_from_blob**: Reads all document files and the traceability matrix workbook from Azure Blob Storage. It stores each document’s binary content and loads the traceability sheet into a pandas DataFrame.
2. **extract_tables_from_file**: Uses the Azure Document Intelligence (Form Recognizer) Layout model to extract tables from each document, returning a list of pandas DataFrames (one per table).
3. **compare_files**: Calls an Azure AI Foundry GPT-4.1 deployment to compare the extracted document tables against the traceability matrix. It returns a JSON array of comparison results (one object per row).

The **main** function (`processFiles`) orchestrates these steps and finally converts the comparison JSON into an HTML table with inline styling, suitable for rendering in Power Apps.

## 📖 Function Details

### grab_from_blob(sas_url, sas_token, container)

* Connects to Blob Storage.
* Lists blobs under `documents/` and `traceability_matrix/`.
* Downloads each file’s binary content.
* Loads Excel sheets via `pandas.read_excel(...)`.
* Returns `(documents, traceability_matrix)`lists.

### extract_tables_from_file(endpoint, key, fcontent)

* Initializes `DocumentIntelligenceClient`.
* Calls `begin_analyze_document(model_id="prebuilt-layout", body=...)`.
* Builds a 2D matrix of cell contents for each table.
* Returns a list of DataFrames.

### compare_files(foundry_url, foundry_key, all_doc_rows, matrix_json)

* Initializes `AzureOpenAI` client.
* Sends a chat completion request to GPT-4.1 with system/user instructions.
* Retries on rate limits using exponential backoff.
* Parses JSON output into Python objects.
* Returns a list of comparison result dicts.

### processFiles(req: HttpRequest)

* Loads `.env` variables.
* Orchestrates the three functions above.
* Builds the final response JSON:
