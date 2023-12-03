from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

raw_dataset = pd.read_csv("author_stats.csv")
dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset = dataset.drop(
    [
        "id",
        "name_of_author",
        "institute",
        "country",
        "institute_ID",
        "citation_count",
        "average_citations",
        "total_papers",
        "paper_freq",
        "author_classification",
    ],
    axis=1,
)  # remove irrelivant data from each author

# sns.pairplot(dataset, y_vars=["h_index"], x_vars=["average_DCM_papers_of_coauthors", "connected_institutes", "connections", "second_order_connections", "average_h_index_of_coauthors"], diag_kind="kde", kind='reg')
sns.pairplot(
    dataset,
    y_vars=["h_index"],
    x_vars=[
        "average_coauthors",
        "connected_institutes",
        "average_h_index_of_coauthors",
        "average_DCM_papers_of_coauthors",
    ],
)

# plt.savefig('figs/variables_with_h.pdf')
# plt.show()
plt.close()

h_indexes = dataset.h_index
second_order_collaborations_per_first_order_collaboration = (
    dataset.second_order_connections / dataset.connections
)
sns.scatterplot(second_order_collaborations_per_first_order_collaboration, h_indexes)
plt.show()
