# Create datasets
For all of these, the shell scripts must be modified to use your own data directories.

**Synthetic (bimodal) dataset**
Generate the dataset:

    ./generate_data_bimodal.sh

**MIND (news) dataset**
Some data is downloaded in the script. Then, we post-process it.

    ./generate_data_MIND.sh


## Extras
For appendix experiments

**Empirical Bayes (Beta-Bernoulli) Example**
Generate the dataset:

	./extras/generate_data_beta.sh

**Comparing TS-PSAR to TS-Action on smaller datasets**
Generate the dataset

	./extras/generate_data_bimodal_small.sh
