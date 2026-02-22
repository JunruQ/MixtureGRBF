library("DEswan")
library(tidyverse)
library(dplyr)

prot_table <- read_csv("input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv")
subtype_stage <- read.csv("output/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/subtype_stage.csv")

merged <- prot_table %>%
  left_join(
    subtype_stage %>% select(PTID, subtype),
    by = c("RID" = "PTID")
  )

# res.DEswan <- DEswan(
#   data.df = merged[, 8:(ncol(merged)-1)],
#   qt = merged$stage,
#   window.center = 50,
#   buckets.size = 1,
#   covariates = merged[, c(3,4)]
# )

res.DEswan <- DEswan(
  data.df = merged[, 926],
  qt = merged$stage,
  covariates = NULL,
  buckets.size = 10
)

res.DEswan.wide.p=reshape.DEswan(res.DEswan,parameter = 1,factor = "qt")
res.DEswan.wide.q=q.DEswan(res.DEswan.wide.p,method="BH")

res.DEswan.wide.q.signif=nsignif.DEswan(res.DEswan.wide.q)
toPlot=res.DEswan.wide.q.signif[1:3,]
x=as.numeric(gsub("X","",colnames(toPlot)))
plot(1, type = "n", xlim=c(min(x,na.rm=T),max(x,na.rm=T)),ylim=c(0,max(toPlot,na.rm=T)),ylab="# significant",xlab="qt")
for(i in 1:nrow(toPlot)){
  lines(x,
        toPlot[i,],type='l',lwd=i)
}
legend("topleft",legend = paste("q<",rownames(toPlot),sep=""),lwd=c(1,2,3))

# prot_table <- read.table(
#   file = "input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv",
#   header = TRUE,
#   sep = ","
# )

# subtype_stage <- read.table(
#   file = "output/
#     ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/subtype_stage.csv",
#   header = TRUE,
#   sep = ","
# )

