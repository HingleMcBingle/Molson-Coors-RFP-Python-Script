
# Azure Document Comparison Function App

This repository contains an Azure Function App that automates the process of extracting tables from project documents, comparing them against a requirements traceability matrix, and returning structured results for downstream consumption (e.g., Power Apps).

---

## 🚀 Overview

1. **grab_from_blob**: Reads all document files and the traceability matrix workbook from Azure Blob Storage. It stores each document’s binary content and loads the traceability sheet into a pandas DataFrame.
2. **extract_tables_from_file**: Uses the Azure Document Intelligence (Form Recognizer) Layout model to extract tables from each document, returning a list of pandas DataFrames (one per table).
3. **compare_files**: Calls an Azure AI Foundry GPT-4.1 deployment to compare the extracted document tables against the traceability matrix. It returns a JSON array of comparison results (one object per row).

The **main** function (`<span>processFiles</span>`) orchestrates these steps and finally converts the comparison JSON into an HTML table with inline styling, suitable for rendering in Power Apps.


## 📖 Function Details

### grab_from_blob(sas_url, sas_token, container)

* Connects to Blob Storage.
* Lists blobs under `<span>documents/</span>` and `<span>traceability_matrix/</span>`.
* Downloads each file’s binary content.
* Loads Excel sheets via `<span>pandas.read_excel(...)</span>`.
* Returns `<span>(documents, traceability_matrix)</span>` lists.

### extract_tables_from_file(endpoint, key, fcontent)

* Initializes `<span>DocumentIntelligenceClient</span>`.
* Calls `<span>begin_analyze_document(model_id="prebuilt-layout", body=...)</span>`.
* Builds a 2D matrix of cell contents for each table.
* Returns a list of DataFrames.

### compare_files(foundry_url, foundry_key, all_doc_rows, matrix_json)

* Initializes `<span>AzureOpenAI</span>` client.
* Sends a chat completion request to GPT-4.1 with system/user instructions.
* Retries on rate limits using exponential backoff.
* Parses JSON output into Python objects.
* Returns a list of comparison result dicts.

### processFiles(req: HttpRequest)

* Loads `<span>.env</span>` variables.
* Orchestrates the three functions above.
* Builds the final response JSON:
