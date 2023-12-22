# Quantifying_clusterness_trajectoriness

This repo provides code to reproduce the results in the research article: Lim & Qiu, "Quantifying the clusterness and trajectoriness of single-cell RNA-seq data".

"PART0_generate_figures.ipynb" provides code for generating 12,000 simulated datasets of cluster-like or trajectory-like geometry, computing the five scoring metrics, generating the geometric landscape, scoring real scRNA-seq datasets, and mapping the real datasets onto the geometric landscape. This notebook reproduces most of the figures in the article. 

"PART1_generate_figures_from_saved.ipynb" reproduces the same set of figures as PART0. The only difference is that PART1 loads pre-generated datasets and pre-computed scores to reproduce the figures. 

"PART2_evaluate_new_data.ipynb" provides examples for how to map a new dataset on to the pre-generated geometric landscape. 

Running these examples requires the data folder. Please follow the readme in this data folder to obtain the data files.

"visualize_benchmarking_datasets.ipynb" provides visiualizations for all 169 benchmarking datasets used in this study.
