import csv
import itertools

import pandas as pd
from pybliometrics.scopus import (
    AbstractRetrieval,
    AuthorRetrieval,
    ScopusSearch,
    config,
)

config["Authentication"]["APIKey"] = "24642bce15a7a6e69757c7a945095542"


def search_for_paper(
    papertitle
):  # searches for papers of given title on Scopus database
    list_of_researcher_ids = []
    try:
        article_search_list = ScopusSearch("TITLE({paper})".format(paper=papertitle))
        # print(article_search_list)
        eid = article_search_list.get_eids()
        eid = eid[0]
        ab = AbstractRetrieval(eid, view="FULL")

        for author in ab.authorgroup:
            list_of_researcher_ids.append(author.auid)
        return list_of_researcher_ids
    except:  # noqa: E722
        return None


def all_combos_of_authors(
    list_of_authors
):  # creates a "links list" for a set of authors
    try:
        combos_list = itertools.combinations(list_of_authors, 2)
        return combos_list
    except:  # noqa: E722
        return None


def add_combos_of_paper_index_to_list(
    paper_index
):  # adds the unique new links to an overall list
    overall_list = []
    paper_index = paper_index
    # print(search_for_paper(paper_titles[paper_index]))

    if all_combos_of_authors(search_for_paper(paper_titles[paper_index])) is not None:
        for combo in all_combos_of_authors(search_for_paper(paper_titles[paper_index])):
            overall_list.append(sorted(list(combo)))
    if overall_list != []:
        return overall_list
    else:
        return None


with open(
    "papers.csv"
) as papers_csv:  # opens csv of all papers. Only the paper titles are used. Creates list of paper titles
    papers = csv.DictReader(papers_csv)
    paper_titles = []
    for row in papers:
        paper_titles.append(row["Title"].replace(".", ""))
    print(len(paper_titles))


successful_papers = 0
list_of_author_lengths = []
papers_searched = 1674  # number of papers to search out of list (useful if only searching the first few papers for debugging)
authors_list = []
freqs = []

tot_coauthorships = []

overall_links_list = []
for paper_index in range(
    papers_searched
):  # searches for papers throughout list, then creates the links list and authors list, as well as counting paper frequency and total coauthors for each author
    print(
        "Currently processing paper ",
        paper_index + 1,
        " of ",
        papers_searched,
        end="\r",
    )
    if add_combos_of_paper_index_to_list(paper_index) is not None:
        successful_papers += 1
        overall_links_list = overall_links_list + add_combos_of_paper_index_to_list(
            paper_index
        )

        list_of_author_lengths.append(
            len(list(set(search_for_paper(paper_titles[paper_index]))))
        )
        for author in list(set(search_for_paper(paper_titles[paper_index]))):
            if authors_list.count(author) == 0:
                authors_list.append(author)
                freqs.append(1)
                tot_coauthorships.append(
                    len(list(set(search_for_paper(paper_titles[paper_index])))) - 1
                )
            else:
                freqs[authors_list.index(author)] += 1
                tot_coauthorships[authors_list.index(author)] += (
                    len(list(set(search_for_paper(paper_titles[paper_index])))) - 1
                )
print()

all_author_details = []
prolific_author_details = []
prolific_authors = []


for author in authors_list:  # gathers relevant data about each author
    print(
        "Currently processing author ",
        authors_list.index(author) + 1,
        " of ",
        len(authors_list),
        end="\r",
    )
    author_details = []

    freq = freqs[authors_list.index(author)]
    co_authorships = tot_coauthorships[authors_list.index(author)]
    author_object = AuthorRetrieval(author_id=str(author))
    h_index = author_object.h_index
    name = author_object.indexed_name
    total_docs = author_object.document_count
    citation_count = author_object.citation_count
    # country = author_object.affiliation_current.country
    av_coauthors = co_authorships / freq

    if (
        author_object.affiliation_current is not None
        and author_object.affiliation_current[0] is not None
        and author_object.affiliation_current[0][8] is not None
    ):
        country = author_object.affiliation_current[0][8]
        if author_object.affiliation_current[0][6] is not None:
            institute = author_object.affiliation_current[0][6]
        else:
            institute = author_object.affiliation_current[0][5]
        institute_ID = author_object.affiliation_current[0][0]
    else:
        country = "NA"

    # for link in overall_links_list:
    #     if link[0]==author or link[1]==author:

    author_details.append(author)
    author_details.append(freq)
    author_details.append(h_index)
    author_details.append(name)
    author_details.append(total_docs)
    author_details.append(citation_count)
    author_details.append(int(citation_count) / int(total_docs))
    author_details.append(country)
    author_details.append(institute)
    author_details.append(institute_ID)
    author_details.append(av_coauthors)
    # print(author_details)
    # author_details.append(country)

    all_author_details.append(author_details)

    if (int(freq) >= 5 or int(freq) >= (0.1 * int(total_docs))) and freq >= 2:
        # if freq>=2:
        prolific_author_details.append(author_details)
        prolific_authors.append(author)
print()

prolific_author_links = []
for pair in overall_links_list:  # ensures links are unique
    if (pair[0] in prolific_authors) and (pair[1] in prolific_authors):
        prolific_author_links.append(pair)


# these 2 functions are useful for calculating information about connected authors, eg the number of countries an author is connected to
def calculate_connected_clusters(
    authorslist, linkslist, author_details_list, author_detail_index
):
    author_connected_clusters = []
    for author in authorslist:
        author_connections = 0
        author_clusters = []
        for link in linkslist:
            if link[0] == author or link[1] == author:
                if link[0] == author:
                    coauthor = link[1]
                elif link[1] == author:
                    coauthor = link[0]
                coauthor_cluster = author_details_list[authorslist.index(coauthor)][
                    author_detail_index
                ]
                if coauthor_cluster in author_clusters:
                    pass
                else:
                    author_connections += 1
                    author_clusters.append(coauthor_cluster)
        author_connected_clusters.append(author_connections)
    return author_connected_clusters


def calculate_av_coauthors(
    authorslist, linkslist, author_details_list, author_detail_index
):
    author_averages = []
    for author in authorslist:
        author_total = 0
        num_coauthors = 0
        for link in linkslist:
            if link[0] == author or link[1] == author:
                if link[0] == author:
                    coauthor = link[1]
                elif link[1] == author:
                    coauthor = link[0]
                coauthor_value = author_details_list[authorslist.index(coauthor)][
                    author_detail_index
                ]
                num_coauthors += 1
                author_total += int(coauthor_value)
        if num_coauthors == 0:
            author_averages.append(0)
        else:
            author_averages.append((author_total / num_coauthors))
    return author_averages


# gathering more author data
country_connections = calculate_connected_clusters(
    prolific_authors, prolific_author_links, prolific_author_details, 7
)
inst_connections = calculate_connected_clusters(
    prolific_authors, prolific_author_links, prolific_author_details, 9
)
av_h_of_coauthors = calculate_av_coauthors(
    prolific_authors, prolific_author_links, prolific_author_details, 2
)
av_freq_of_coauthors = calculate_av_coauthors(
    prolific_authors, prolific_author_links, prolific_author_details, 1
)

for author_index in range(
    len(prolific_authors)
):  # saving more author data to the author details list
    prolific_author_details[author_index].append(country_connections[author_index])
    prolific_author_details[author_index].append(inst_connections[author_index])
    prolific_author_details[author_index].append(av_h_of_coauthors[author_index])
    prolific_author_details[author_index].append(av_freq_of_coauthors[author_index])

for author_details in prolific_author_details:  # classifies authors. This is useful for a predictive program which qualitatively groups authors. The data made here is not actually used, but was explored (see discussion)
    if int(author_details[2]) >= 40 and int(author_details[1]) >= 10:
        author_details.append(0)
    elif int(author_details[2]) >= 30 and int(author_details[1]) >= 5:
        author_details.append(1)
    elif int(author_details[2]) >= 10 or int(author_details[5]) >= 200:
        author_details.append(2)
    else:
        author_details.append(3)

with open(
    "readouts/successful_papers_readout.txt", "w"
) as printout:  # records how many of the papers were sucessfully searched
    printout_message = (
        str(successful_papers)
        + " papers of "
        + str(papers_searched)
        + " successfully analysed"
    )
    printout.write(printout_message)

# processes data of prolific authors, relevant to research, for network analysis. The non-prolific authors are disregarded
prolific_links_df = pd.DataFrame(prolific_author_links, columns=["from", "to"])
prolific_links_df.drop_duplicates(subset=["from", "to"], inplace=True)
prolific_authors_df = pd.DataFrame(
    prolific_author_details,
    columns=[
        "author_ID",
        "paper_freq",
        "h_index",
        "name_of_author",
        "total_papers",
        "citation_count",
        "average_citations",
        "country",
        "institute",
        "institute_ID",
        "average_coauthors",
        "connected_countries",
        "connected_institutes",
        "average_h_index_of_coauthors",
        "average_DCM_papers_of_coauthors",
        "author_classification",
    ],
)
print(prolific_authors_df)
prolific_links_df.to_csv("Python_links_prolific.csv", index=False)
prolific_authors_df.to_csv("Python_authors_prolific.csv", index=False)
