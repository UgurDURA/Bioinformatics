install.packages("rrBLUP")
install.packages("corrplot")

library(rrBLUP)
library(corrplot)

head(pheno)
str(pheno)

hist(pheno$Yield, col = "black", xlab = "Yield", ylab = "Frequency",
     border = "white", breaks = 10, main = "Yield Histogram")

 
shapiro.test(pheno$Yield)

boxplot.yield <- boxplot(pheno$Yield, xlab = "BoxPlot", ylab = "Yield",
                         ylim = c(4000,9000))

outliers <- boxplot.yield$out; outliers #10 outliers

pheno <- pheno[-which(pheno$Yield%in%outliers),]

shapiro.test(pheno$Yield)


filter.fun <- function(geno,IM,MM,H){
  #Remove individuals with more than a % missing data
  individual.missing <- apply(geno,1,function(x){
    return(length(which(is.na(x)))/ncol(geno))
  })
  #length(which(individual.missing>0.40)) #will tell you how many 
  #individulas needs to be removed with 20% missing.
  #Remove markers with % missing data
  marker.missing <- apply(geno,2,function(x)
  {return(length(which(is.na(x)))/nrow(geno))
    
  })
  length(which(marker.missing>0.6))
  #Remove markers herteozygous calls more than %. 
  heteroz <- apply(geno,1,function(x){
    return(length(which(x==0))/length(!is.na(x)))
  })
  
  filter1 <- geno[which(individual.missing<IM),which(marker.missing<MM)]
  filter2 <- filter1[,(heteroz<H)]
  return(filter2)
}

geno.filtered <- filter.fun(geno[,1:3629],0.4,0.60,0.02)
geno.filtered[1:5,1:5];dim(geno.filtered)

library(rrBLUP)
Imputation <- A.mat(geno.filtered,impute.method="EM",return.imputed=T,min.MAF=0.05)

K.mat <- Imputation$A ### KINSHIP matrix
geno.gwas <- Imputation$imputed #NEW geno data.
geno.gwas[1:5,1:5]## view geno
K.mat[1:5,1:5]## view Kinship

################***CHECKING POPULATION STRUCTURE EFFECTS***###############
## Principal components analysis
dev.off()
geno.scale <- scale(geno.gwas,center=T,scale=F) # Data needs to be center.
svdgeno <- svd(geno.scale) 
PCA <- geno.scale%*%svdgeno$v #Principal components
PCA[1:5,1:5]

plot(round((svdgeno$d)^2/sum((svdgeno$d)^2),d=7)[1:10],type="o",main="Screeplot",xlab="PCAs",ylab="% variance")

PCA1 <- 100*round((svdgeno$d[1])^2/sum((svdgeno$d)^2),d=3); PCA1
PCA2 <- 100*round((svdgeno$d[2])^2/sum((svdgeno$d)^2),d=3); PCA2
PCA3 <- 100*round((svdgeno$d[3])^2/sum((svdgeno$d)^2),d=3); PCA3
PCA4 <- 100*round((svdgeno$d[3])^2/sum((svdgeno$d)^2),d=3); PCA4
plot(PCA[,1],PCA[,2],xlab=paste("Pcomp:",PCA1,"%",sep=""),ylab=paste("Pcomp:",PCA2,"%",sep=""),pch=20,cex=0.7)

Eucl <- dist(geno.gwas) ###Euclinean distance
fit <- hclust(Eucl,method="ward.D2")### Ward criterion makes clusters with same size.
groups2 <- cutree(fit,k=6) ### Selecting two clusters.
table(groups2)

plot(PCA[,1],PCA[,2],xlab=paste("Pcomp:",PCA1,"%",sep=""),ylab=paste("Pcomp:",PCA2,"%",sep=""),pch=20,cex=0.7,col=groups2)
legend("bottomright",c("Subpop1: 244","Subpop2: 84"),pch=20,col=(c("black","red")),lty=0,bty="n",cex=1)


################***MATCHING PHENOTYPE AND GENOTYPE***###############

pheno=pheno[pheno$GID%in%rownames(geno.gwas),]
pheno$GID<-factor(as.character(pheno$GID), levels=rownames(geno.gwas)) #to assure same levels on both files
pheno <- pheno[order(pheno$GID),]
##Creating file for GWAS function from rrBLUP package
X<-model.matrix(~-1+ENV, data=pheno)
pheno.gwas <- data.frame(GID=pheno$GID,X,Yield=pheno$Yield)
head(pheno.gwas)

geno.gwas <- geno.gwas[rownames(geno.gwas)%in%pheno.gwas$GID,]
pheno.gwas <- pheno.gwas[pheno.gwas$GID%in%rownames(geno.gwas),]
geno.gwas <- geno.gwas[rownames(geno.gwas)%in%rownames(K.mat),]
K.mat <- K.mat[rownames(K.mat)%in%rownames(geno.gwas),colnames(K.mat)%in%rownames(geno.gwas)]
pheno.gwas <- pheno.gwas[pheno.gwas$GID%in%rownames(K.mat),]


################***MATCHING GENOTYPE AND MAP***###############
geno.gwas<-geno.gwas[,match(map$Markers,colnames(geno.gwas))]
head(map)
geno.gwas <- geno.gwas[,colnames(geno.gwas)%in%map$Markers]
map <- map[map$Markers%in%colnames(geno.gwas),]
geno.gwas2<- data.frame(mark=colnames(geno.gwas),chr=map$chrom,loc=map$loc,t(geno.gwas))
dim(geno.gwas2)
colnames(geno.gwas2)[4:ncol(geno.gwas2)] <- rownames(geno.gwas)

head(pheno.gwas)
geno.gwas2[1:6,1:6]
K.mat[1:5,1:5]


#####################################***ANALYSIS***################################################
gwasresults<-GWAS(pheno.gwas,geno.gwas2, fixed=colnames(pheno.gwas)[2:5], K=NULL, plot=T,n.PC=0)  # GWAS analysis without accounting PCA
gwasresults2<-GWAS(pheno.gwas,geno.gwas2, fixed=colnames(pheno.gwas)[2:5], K=NULL, plot=T,n.PC=6) # Only accounts PCA
gwasresults3<-GWAS(pheno.gwas,geno.gwas2, fixed=colnames(pheno.gwas)[2:5], K=K.mat, plot=T,n.PC=0) # Accounting Kmat PCA = 0
gwasresults4<-GWAS(pheno.gwas,geno.gwas2, fixed=colnames(pheno.gwas)[2:5], K=K.mat, plot=T,n.PC = 6) # Accounting Kmat with PCA = 6 (Population Structure)

###################################################################################
####################*** QQ-MANHATTAN and CORRELATION PLOTS*****####################
###################################################################################

str(gwasresults)
str(gwasresults2)
str(gwasresults3)
str(gwasresults4)
#First 3 columns are just the information from markers and map.
#Fouth and next columns are the results form GWAS. Those values are already
#the  -log10 pvalues, so no more transformation needs to be done to plot them. 



###################################################################################
#################################*** QQ PLOT*****##################################
###################################################################################

 
par(mfrow=c(2,2))
N <- length(gwasresults$Yield)
expected.logvalues <- sort( -log10( c(1:N) * (1/N) ) )
observed.logvalues <- sort( gwasresults$Yield)

plot(expected.logvalues , observed.logvalues, main="Naïve model(K=NULL,n.PC=0)", 
     xlab="expected -log pvalue ", 
     ylab="observed -log p-values",col.main="blue",col="coral1",pch=20)
abline(0,1,lwd=3,col="black")


N1 <- length(gwasresults2$Yield)
expected.logvalues1 <- sort( -log10( c(1:N1) * (1/N1) ) )
observed.logvalues1 <- sort( gwasresults2$Yield)

plot(expected.logvalues1 , observed.logvalues1, main="Q model (K=NULL,n.PC=6)", 
     xlab="expected -log pvalue ", 
     ylab="observed -log p-values",col.main="blue",col="coral1",pch=20)
abline(0,1,lwd=2,col="black")


N2 <- length(gwasresults3$Yield)
expected.logvalues2 <- sort( -log10( c(1:N2) * (1/N2) ) )
observed.logvalues2 <- sort( gwasresults3$Yield)

plot(expected.logvalues2 , observed.logvalues2, main="K model (K=Kmat,n.PC=0)", 
     xlab="expected -log pvalue ", 
     ylab="observed -log p-values",col.main="blue",col="coral1",pch=20)
abline(0,1,lwd=2,col="black")

N3 <- length(gwasresults4$Yield)
expected.logvalues3 <- sort( -log10( c(1:N3) * (1/N3) ) )
observed.logvalues3 <- sort( gwasresults4$Yield)

plot(expected.logvalues3 , observed.logvalues3, main="Q+K model (K.mat,n.PC=6)", 
     xlab="expected -log pvalue ", 
     ylab="observed -log p-values",col.main="blue",col="coral1",pch=20)
abline(0,1,lwd=2,col="black")


###################################################################################
#################################*** MANHATTAN PLOT*****###########################
###################################################################################
#False Discovery Rate Function

FDR<-function(pvals, FDR){
  pvalss<-sort(pvals, decreasing=F)
  m=length(pvalss)
  cutoffs<-((1:m)/m)*FDR
  logicvec<-pvalss<=cutoffs
  postrue<-which(logicvec)
  print(postrue)
  k<-max(c(postrue,0))
  cutoff<-(((0:m)/m)*FDR)[k+1]
  return(cutoff)
}

alpha_bonferroni=-log10(0.05/length(gwasresults$Yield)) ###This is Bonferroni correcton
alpha_FDR_Yield <- -log10(FDR(10^(-gwasresults$Yield),0.05))## This is FDR cut off

#################################*** MANHATTAN PLOT*****###########################

dev.off()
 
plot(gwasresults$Yield,ylab="-log10.pvalue",
     main="Naïve model (K=NULL,n.PC=0)",xaxt="n",xlab="Position",ylim=c(0,14))
axis(1,at=c(1:length(unique(gwasresults$chr))),labels=unique(gwasresults$chr))
axis(1,at=c(0,440,880,1320,1760))
abline(a=NULL,b=NULL,h=alpha_bonferroni,col="blue",lwd=2)
abline(a=NULL,b=NULL,h=alpha_FDR_Yield,col="red",lwd=2,lty=2)
legend(1,13.5, c("Bonferroni","FDR") , 
       lty=1, col=c('red', 'blue'), bty='n', cex=1,lwd=2)

plot(gwasresults2$Yield,ylim=c(0,14),ylab="-log10.pvalue",
     main="Q model (K=NULL,n.PC=6)",xaxt="n",xlab="Position")
axis(1,at=c(0,440,880,1320,1760))
abline(a=NULL,b=NULL,h=alpha_bonferroni,col="blue",lwd=2)
abline(a=NULL,b=NULL,h=alpha_FDR_Yield,col="red",lwd=2,lty=2)
legend(1.5,13.5, c("Bonferroni","FDR") , 
       lty=1, col=c('red', 'blue'), bty='n', cex=1,lwd=2)

plot(gwasresults3$Yield,col=gwasresults3$chr,ylim=c(0,14),ylab="-log10.pvalue",
     main="K model (K=K.mat,n.PC=0)",xaxt="n",xlab="Position")
axis(1,at=c(0,440,880,1320,1760))
abline(a=NULL,b=NULL,h=alpha_bonferroni,col="blue",lwd=2)
abline(a=NULL,b=NULL,h=alpha_FDR_Yield,col="red",lwd=2,lty=2)
legend(1.5,13.5, c("Bonferroni","FDR") , 
       lty=1, col=c('red', 'blue'), bty='n', cex=1,lwd=2)

plot(gwasresults4$Yield,col=gwasresults4$chr,ylim=c(0,14),ylab="-log10.pvalue",
     main="Q+K model (K=K.mat,n.PC=6)",xaxt="n",xlab="Position")
axis(1,at=c(0,440,880,1320,1760))
abline(a=NULL,b=NULL,h=alpha_bonferroni,col="blue",lwd=2)
abline(a=NULL,b=NULL,h=alpha_FDR_Yield,col="red",lwd=2,lty=2)##FDR gives inf for Yield
legend(1,13.5, c("Bonferroni","FDR") , 
       lty=1, col=c('red', 'blue'), bty='n', cex=1,lwd=2)
############################*** WHICH ARE HITS?*****###########################

which(gwasresults$Yield>alpha_bonferroni)
which(gwasresults$Yield>alpha_FDR_Yield)
which(gwasresults2$Yield>alpha_bonferroni)
which(gwasresults2$Yield>alpha_FDR_Yield)
which(gwasresults3$Yield>alpha_bonferroni)
which(gwasresults3$Yield>alpha_FDR_Yield)
which(gwasresults4$Yield>alpha_bonferroni)
which(gwasresults4$Yield>alpha_FDR_Yield)

markers.gwasresults4.fdr<- geno.gwas[,c(45 ,  53 ,  56 ,  57 ,1054 ,1427 ,1428)]#gwasresults3 and 4 have same hits.
markers.gwasresults2.fdr <- geno.gwas[,c( 45  , 53  , 56 ,  57, 1054 ,1427 ,1428)]

###################################################################################
##############################*** CORRELATION PLOT*****############################
###################################################################################

library(corrplot)
corr_sign <- cor(markers.gwasresults4.fdr,use="complete.obs") 

corrplot(corr_sign, order="hclust", method="pie", tl.pos="lt", type="upper",        
         tl.col="black", tl.cex=0.8, tl.srt=55, 
         sig.level=0.90,cl.length=21,insig = "blank")     
mtext("Correlation Significant hits",outer=TRUE,line=1) 





