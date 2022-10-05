# External datasets

Some experiments rely on third-party datasets that are publicly available
online.

- PBC2: `DOC-10026794`, available at
  <https://www.mayo.edu/research/documents/pbcseqhtml/doc-10027141>
- AIDS: `aids.rda` available at
  <https://github.com/drizopoulos/JM/tree/master/data>

## Preparing the data

For the PBC2 dataset, simply rename the file to include a `.dat` extension

    mv DOC-10026794 DOC-10026794.dat

For the AIDS dataset, you will need to use R to transform the R data file into
a CSV.

    load("aids.rda")
    write.csv(aids, "aids.csv", row.names = TRUE)

The notebooks use the resulting `aids.csv` file.
