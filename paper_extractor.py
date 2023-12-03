import requests

# Define the base URL for the PubMed API
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Define the search term and maximum number of returns
search_term = "degenerative cervical myelopathy"
retmax = 100

# Define the search URL
search_url = (
    f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmode=json&retmax={retmax}"
)

# Send a GET request to the PubMed API
response = requests.get(search_url)


# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    results = response.json()

    # Extract the list of PubMed IDs (PMIDs)
    pmids = results["esearchresult"]["idlist"]
    count = results["esearchresult"]["count"]
    print(count)

    # Join the list of PMIDs into a comma-separated string
    id_string = ",".join(pmids)

    # Request the summary for all PMIDs
    summary_url = f"{base_url}esummary.fcgi?db=pubmed&id={id_string}&retmode=json"
    summary_response = requests.get(summary_url)

    if summary_response.status_code == 200:
        summary_results = summary_response.json()
        # Extract and print the titles
        with open(
            "./biology_notes/DCM_network_analysis_repo/code/new_papers.csv", "w"
        ) as f:
            headings = "Journal, Year, Authors, Title, Paper Ref, Country\n"
            f.write(headings)
        for pmid in pmids:
            paper = summary_results["result"][pmid]
            journal = paper["fulljournalname"]
            year = paper["epubdate"]
            authors = [a["name"] for a in paper["authors"]]
            title = paper["title"]
            # country = summary_results["locationlabel"]
            line = f"{journal}, {year}, {authors}, {title}, {pmid}\n"

            with open(
                "./biology_notes/DCM_network_analysis_repo/code/new_papers.csv", "a"
            ) as f:
                f.write(line)
    else:
        print(
            f"Failed to fetch summary details with status code {summary_response.status_code}"
        )

else:
    print(f"Request failed with status code {response.status_code}")
