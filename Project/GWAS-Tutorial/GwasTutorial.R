install.packages("BiocManager")
BiocManager::install('snpStats')

library(snpStats)

obj <- read.plink('data/GWAStutorial')

plot(row.summary(obj$genotypes)[c(1,3)])
