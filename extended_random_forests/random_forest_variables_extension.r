setwd("/Users/dupe/projects/princeton/LTPMuchlinski_extension/replication")
getwd()

# data for prediction
data=read.csv(file="SambanisImp.csv")

library(randomForest) #for random forests
library(caret) # for CV folds and data splitting
library(ROCR) # for diagnostics and ROC plots/stats
library(pROC) # same as ROCR
library(stepPlr) # Firth’s logit implemented thru caret library
library(doMC) # for using multiple processor cores
library(xtable) # for writing Table 1 in Latex

###Use only the 88 variables specified in Sambanis (2006) Appendix###
data.full<-data[,c("warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4",
                   "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
                   "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
                   "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
                   "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
                   "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
                   "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper",
                   "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
                   "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
                   "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
                   "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
                   "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc",
                   "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
                   "warhist", "xconst")]

###Convert DV into Factor with names for Caret Library###
data.full$warstds<-factor(
  data.full$warstds,
  levels=c(0,1),
  labels=c("peace", "war"))

# distribute workload over multiple cores for faster computation
registerDoMC(cores=7)
set.seed(666)

tc<-trainControl(method="cv",
                 number=10,
                 summaryFunction=twoClassSummary,
                 classProb=T,
                 savePredictions = T)

### Random Forest with all variables (Muchlinksi original) ###
model.rf<-train(as.factor(warstds)~.,
                metric="ROC", method="rf",
                sampsize=c(30,90),
                importance=T,
                proximity=F, ntree=1000,
                trControl=tc, data=data.full)
model.rf

# EXTENSION: Train a random forest with only the top 9 variables 
# ranked by mean decrease in gini score. These are the top variables identified
# by Muchlinksi et al
model.rftop9<-train(as.factor(warstds)~gdpgrowth+ln_gdpen+life+infant+lmtnest
                    +pol4sq+lpopns+trade+geo1,
                    metric="ROC", method="rf", sampsize=c(30,90),importance=T,
                    proximity=F, ntree=1000,trControl=tc,data=data.full)
model.rftop9

# EXTENSION: Random forest with only gdp variables, the top 2 variables
#            ranked by mean decrease in Gini score
model.rf_gdp<-train(as.factor(warstds)~gdpgrowth+ln_gdpen,
                    metric="ROC", method="rf", sampsize=c(30,90),importance=T,
                    proximity=F, ntree=1000,trControl=tc,data=data.full)
model.rf_gdp

# EXTENSION: Train random forest with top 9 variables ranked by mean decrease in accuracy
#            Note: drace was in this top 9, but it's not in the sambanis 88 variables,
#            so we skip it to maintain our ability to compare against the Muchlinski
#            random forest
model.rf_topvars_accuracy<-train(as.factor(warstds)~drel+milper+warhist+second+dlang+dem
                                 +gdpgrowth+etdo4590+lmtnest,
                                 metric="ROC", method="rf", sampsize=c(30,90),importance=T,
                                 proximity=F, ntree=1000,trControl=tc,data=data.full)
model.rf_topvars_accuracy

##Random Forests on Amelia Imputed Data for Variable Importance Plot###
##Data Imputed only for Theoretically Important Variables### Done to analyze variable
# importance
data2<-read.csv(file="Amelia.Imp3.csv") ###
myvars <- names(data2) %in% c("X", "country", "year", "atwards")
newdata <- data2[!myvars]
RF.out.am<-randomForest(as.factor(warstds)~.,sampsize=c(30, 90),
                        importance=T, proximity=F, ntree=1000, confusion=T, err.rate=T, data=newdata)

### EXTENSION: Variable Importance Plot based on Mean Decrease in Accuracy
varImpPlot(RF.out.am, sort=T, type=1, 
main="Random Forest Variable Importance by Mean Decrease in Accuracy", 
n.var=20)
importance(RF.out.am)

### ROC Plots for Different Models ###
library(ROCR)
attach(data.full)

RF.1.pred<-predict(model.rf$finalModel, type="prob")
RF.1.pred<-as.data.frame(RF.1.pred)

# EXTENSION: Predictions using the random forests with different variables
RFtop9.pred<-predict(model.rftop9$finalModel, type="prob")
RFtop9.pred<-as.data.frame(RFtop9.pred)
rf_gdp.pred<-predict(model.rf_gdp$finalModel, type="prob")
rf_gdp.pred<-as.data.frame(rf_gdp.pred)
rf_topvars_accuracy.pred<-predict(model.rf_topvars_accuracy$finalModel, type="prob")
rf_topvars_accuracy.pred<-as.data.frame(rf_topvars_accuracy.pred)

pred.RF.1<-prediction(RF.1.pred$war, data.full$warstds)
perf.RF.1<-performance(pred.RF.1, "tpr", "fpr")
perf_auc.RF.1<-performance(pred.RF.1, "auc")

pred.RFtop9<-prediction(RFtop9.pred$war, data.full$warstds)
perf.RFtop9<-performance(pred.RFtop9, "tpr", "fpr")
perf_auc.RFtop9<-performance(pred.RFtop9, "auc")

pred.rf_gdp<-prediction(rf_gdp.pred$war, data.full$warstds)
perf.rf_gdp<-performance(pred.rf_gdp, "tpr", "fpr")
perf_auc.rf_gdp<-performance(pred.rf_gdp, "auc")

pred.rf_topvars_accuracy<-prediction(rf_topvars_accuracy.pred$war, data.full$warstds)
perf.rf_topvars_accuracy<-performance(pred.rf_topvars_accuracy, "tpr", "fpr")
perf_auc.rf_topvars_accuracy<-performance(pred.rf_topvars_accuracy, "auc")

plot(perf.RF.1, main = "Random Forest Models ROC Plots")
plot(perf.RFtop9, add=T, lty=2)
plot(perf.rf_gdp, add=T, lty=3)
plot(perf.rf_topvars_accuracy, add=T, lty=4)
#plot(perf.CH_allvars, add=T, lty=4)
legend(0.32, 0.25, c("Random Forest 0.91", "Random Forest Top 9 Variables by Gini Score 0.87", 
                     "Random Forest GDP variables 0.76", "Random Forest Top 9 Variables by Mean Accuracy Decrease 0.84"), 
       lty=c(1,2,3,4), bty="n",
       cex = .75)

###Separation Plots###
library(separationplot)

## Transform DV back to 0,1 values for separation plots.
data.full$warstds<-factor(
  data.full$warstds,
  levels=c("peace","war"),
  labels=c(0, 1))

#transform actual observations into vector for separation plots.
Warstds<-as.vector(data.full$warstds)

###Corrected Separation Plots###
### The corrections are the extraction of the fitted values for the logistic regression
#models
### from caret CV procedure.
separationplot(RF.1.pred$war, Warstds, type = "line", line = T, lwd2=1,
               show.expected=T,
               heading="Random Forests All Variables", height=2.5, col0="white", col1="black")

separationplot(RFtop9.pred$war, Warstds, type = "line", line = T, lwd2=1,
               show.expected=T,
               heading="Random Forests Top Variables Gini Score", height=2.5, col0="white", col1="black")

separationplot(rf_gdp.pred$war, Warstds, type = "line", line = T, lwd2=1,
               show.expected=T,
               heading="Random Forest GDP variables", height=2.5, col0="white", col1="black")

separationplot(rf_topvars_accuracy.pred$war, Warstds, type = "line", line = T, lwd2=1,
               show.expected=T,
               heading="Random Forest Top Variables Mean Decrease Accuracy", height=2.5, col0="white", col1="black")


# Generate out of sample predictions on test data
set.seed(425)

### Dataset for imputation.
data_imp<-read.csv(file="data_full.csv")

# Imputation procedure.
# This is the imputation procedure we originally used to impute this data.
rf.imp<-rfImpute(data_imp, as.factor(data_imp$warstds), iter=5, ntree=1000)

###Out of Sample Data ###
# Subsetting imputed data.
mena<-subset(rf.imp, rf.imp$year > 2000)

rf.pred<-predict(model.rf, newdata=mena, type="prob")
rf.pred<-as.data.frame(rf.pred)
pred.RF.1<-prediction(rf.pred$war, mena$`as.factor(data_imp$warstds)`)
perf.RF.1<-performance(pred.RF.1, "tpr", "fpr")
perf.RF.1<-performance(pred.RF.1, "auc")

# EXTENSION: random forest variations
RFtop9.pred<-predict(model.rftop9, newdata=mena, type="prob")
RFtop9.pred<-as.data.frame(RFtop9.pred)
pred.RFtop9<-prediction(RFtop9.pred$war, mena$`as.factor(data_imp$warstds`)
perf.RFtop9<-performance(pred.RFtop9, "tpr", "fpr")
perf.RFtop9<-performance(pred.RFtop9, "auc")

rf_gdp.pred<-predict(model.rf_gdp, newdata=mena, type="prob")
rf_gdp.pred<-as.data.frame(rf_gdp.pred)
pred.rf_gdp<-prediction(rf_gdp.pred$war, mena$`as.factor(data_imp$warstds`)
perf.rf_gdp<-performance(pred.rf_gdp, "tpr", "fpr")
perf.rf_gdp<-performance(pred.rf_gdp, "auc")

rf_topvars_accuracy.pred<-predict(model.rf_topvars_accuracy, newdata=mena, type="prob")
rf_topvars_accuracy.pred<-as.data.frame(rf_topvars_accuracy.pred)
pred.rf_topvars_accuracy<-prediction(rf_topvars_accuracy.pred$war, mena$`as.factor(data_imp$warstds`)
perf.rf_topvars_accuracy<-performance(pred.rf_topvars_accuracy, "tpr", "fpr")
perf.rf_topvars_accuracy<-performance(pred.rf_topvars_accuracy, "auc")

### Save Imputed Data. ###
predictions<-cbind(mena$cowcode, mena$year, mena$warstds,
                   rf.pred[,2], RFtop9.pred[,2], rf_gdp.pred[,2], 
                   rf_topvars_accuracy.pred[,2])

### Write column headings for the out of sample data. ###
colnames(predictions)<-c("COWcode", 
                         "year", 
                         "CW_Onset",
                         "Random Forest All Variables", 
                         "Random Forest Top 9 Variables Gini Score", 
                         "Random Forest GDP Variables",
                         "Random Forest Top 9 Variables Mean Accurracy Decrease")

### Save predictions as data frame for ordering the columns.
predictions<-as.data.frame(predictions)

### Table 1 Results, ordered by Onset (decreasing), and year (increasing) in R rather
# than excel.
Onset_table<-predictions[order(-predictions$CW_Onset, predictions$year),]

### Rows 1-19 of the above go in Table 1. ###
Onset_table_1thru19<-head(Onset_table, n=19)

### Here's the code for Table 1 in Latex. ###
xtable(Onset_table_1thru19)

### Write the .csv file for all predictions to check against the Latex code for Table 1.
### Sort the csv same way as the Latex table - CW_Onset (decreasing), then by year
# (increasing).
write.csv(predictions, file="out_of_sample_random_forest_predictions_extension.csv")