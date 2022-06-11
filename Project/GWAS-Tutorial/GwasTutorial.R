install.packages("BiocManager")
BiocManager::install('snpStats')

library(snpStats)

obj <- read.plink('data/GWAStutorial')

#there are 1401 points, one for each subject) of whether the call rate 
# of genotype calls that are non-missing) is related to the heterozygosity rate 
#of loci that are called AB, as opposed to AA or BB):
plot(row.summary(obj$genotypes)[c(1,3)]) 

#-----------------------------------------------------------------------
x <- as(obj$genotypes[,143], 'numeric')
fisher.test(drop(x), GWAStutorial_clinical$CAD)

all.equal(rownames(x), as.character(GWAStutorial_clinical$FamID))

data(for.exercise)
snps.10

rs <- row.summary(obj$genotypes)
cs <- col.summary(obj$genotypes)
ggbox <- function (X, xlab = "ind", ylab = "values") {
  if (!is.data.frame(X)) X <- as.data.frame(X)
  ggplot2::ggplot(utils::stack(X), ggplot2::aes_string("ind", 
                                                       "values")) + ggplot2::geom_boxplot() + ggplot2::xlab(xlab) + 
    ggplot2::ylab(ylab)
}

table(obj$map$chromosome)
ggbox(rs$Call.rate, 'Individuals', 'Call rate')
ggbox(cs$Call.rate, 'SNPs', 'Call rate')

#Minor Allel Frequency (MAF)
(Tab <- table(as(obj$genotypes[,143], 'numeric')))

(2*Tab[1] + Tab[2]) / (2*sum(Tab)) #Same functions with the belown MAF function
cs[143,]$MAF

hist(cs$MAF, breaks=seq(0, 0.5, 0.01), border='white', col='gray', las=1)

table(cs$MAF < 0.001)
table(as(obj$genotypes[,62], 'numeric'))

table(cs$MAF == 0)   

obj$map[head(which(cs$MAF==0)),] 

table(obj$fam$sex, GWAStutorial_clinical$sex)



ggbox(cs$z.HWE) 
table(as(obj$genotypes[,which.max(cs$z.HWE)], 'character'))

ggbox(rs$Heterozygosity)

keep <- cs$MAF > 0.001 &
  cs$Call.rate > 0.9 &
  abs(cs$z.HWE) < 6.5
table(keep)

(obj$genotypes <- obj$genotypes[, keep])
obj$map <- obj$map[keep, ]

saveRDS(obj, 'data/gwas-qc.rds')

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("SNPRelate")
library(snpStats)
library(SNPRelate)
library(data.table)
library(magrittr)

obj <- readRDS('data/gwas-qc.rds')
obj$genotypes

dim(obj$map)
cs <- col.summary(obj$genotypes)

table(cs$Call.rate == 1)

?snp.imputation

rules <- snpStats::snp.imputation(obj$genotypes, minA=0)
rules_imputed_numeric <- impute.snps(rules, obj$genotypes, as.numeric = TRUE)

call_rates <- sapply(X = 1:ncol(rules_imputed_numeric),
                     FUN = function(x){sum(!is.na(rules_imputed_numeric[,x]))/nrow(rules_imputed_numeric)})

sum(call_rates >= 0.5) 

to_impute <- which(call_rates < 1)

impute_mean <- function(j){
  # identify missing values in a numeric vector
  miss_idx <- which(is.na(j)) 
  # replace missing values with that SNP mean
  j[miss_idx] <- mean(j, na.rm = TRUE) 
  
  return(j)
}

fully_imputed_numeric <- rules_imputed_numeric

fully_imputed_numeric[,to_impute] <- apply(X = rules_imputed_numeric[,to_impute],
                                           MARGIN = 2,
                                           FUN = impute_mean)

rm(rules_imputed_numeric)

# Load our packages  (same ones mentioned in the data module)
# two pacakges are from Bioconductor 
library(snpStats) 
library(SNPRelate)

# all other packages are on CRAN 
library(data.table)
library(dplyr)
library(magrittr)
library(ggplot2)
library(ncvreg)
library(RSpectra)

# register cores for parallel processing 
library(doSNOW)
registerDoSNOW(makeCluster(4))

smaller_data <- read.delim("https://s3.amazonaws.com/pbreheny-data-sets/admixture.txt")
# see what it looks like 
smaller_data[1:5, 1:5]
#               Race Snp1 Snp2 Snp3 Snp4
# 1 African American    0    0    0    0
# 2 African American    0    0    0    0
# 3 African American    0    0    0    0
# 4 African American    0    0    0    0
# 5 African American    1    0    0    1
# assign the 'race' variable 
race <- smaller_data$Race
# create a matrix with only SNP data 
all_SNPs <- as.matrix(smaller_data[,-1])
# filter out monomorphic SNPs
polymorphic <- apply(all_SNPs, 2, sd) != 0
SNPs<- all_SNPs[,polymorphic] 

# look at the resulting matrix dimensions
dim(SNPs)
# [1] 197  98
# see what the data set looks like in terms of racial categories 
table(race)

pca <- prcomp(SNPs, center = TRUE, scale = TRUE)
# look at the results 
pca$x[1:5, 1:5]
#            PC1        PC2        PC3        PC4        PC5
# [1,] 1.6652335 -0.3909691 -0.7799540 -2.3234440 -2.0802964
# [2,] 1.3048975 -2.2997440 -0.4810622  1.9478506  1.6806863
# [3,] 3.2633748 -0.1948454 -0.6283735  2.7211461  0.9575737
# [4,] 0.5371343 -1.5322968  0.6324284 -0.4261897  2.3696399
# [5,] 3.3810463 -1.2142128 -1.8344268 -0.8499357 -1.6181324
# plot the top 10 PCs in a scree plot 
plot(x = 1:10,
     y = 100 * proportions(pca$sdev[1:10]^2),
     type = 'b',
     ylab = 'Proportion of variance explained',
     xlab = 'PC',
     main = 'Example Scree Plot')

pca_dat <- data.frame(race = race, PC1 = pca$x[,1], PC2 = pca$x[, 2])
pca_plot <- ggplot(pca_dat, aes(x = PC1, y = PC2, col = race)) +
  geom_point() +
  coord_fixed()
plot(pca_plot)

n <- 197
p <- 98
X <- matrix(rnorm(n * p), n, p)
pca_rand <- prcomp(X, center = TRUE, scale = TRUE)
rand_dat <- data.frame(PC1 = pca_rand$x[,1], PC2 = pca_rand$x[, 2])
rand_plot <- ggplot(rand_dat,
                    aes(x = PC1, y = PC2)) +
  geom_point() +
  coord_fixed()
plot(rand_plot)


X <- readRDS(file = "data/fully_imputed_numeric.rds")





#--------------------------------------------




clinical <- fread("data/GWAStutorial_clinical.csv")
obj <- readRDS('data/gwas-qc.rds')
(bim <- fread('data/GWAStutorial.bim'))
obj$genotypes
dim(obj$map)

assoc_test <- snp.rhs.tests(
  clinical$CAD ~ clinical$sex + clinical$age, 
  family   = "binomial",
  data     = obj$fam,
  snp.data = obj$genotypes
)

assoc_p_vals <- p.value(assoc_test)
