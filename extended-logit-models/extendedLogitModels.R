setwd("/Users/tara/Documents/Princeton/Academics/Fall 2020/COS 597E/Assignment 2/Muchlinski Replication Materials/dataverse_files") 
getwd()

# data for prediction 
data=read.csv(file="../dataverse_files/SambanisImp.csv")

library(randomForest) #for random forests 
library(caret) # for CV folds and data splitting 
library(ROCR) # for diagnostics and ROC plots/stats 
library(pROC) # same as ROCR
library(stepPlr) # Firth’s logit implemented thru caret library 
library(doMC) # for using multiple processor cores
library(xtable)

###Use only the 88 variables specified in Sambanis (2006) Appendix### 
data.full<-data[,c("warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4", "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
                   "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
                   "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
                   "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
                   "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
                   "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper", "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
                   "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
                   "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
                   "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
                   "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc", "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
                   "warhist", "xconst")]

###Convert DV into Factor with names for Caret Library###
data.full$warstds<-factor(
  data.full$warstds,
  levels=c(0,1),
  labels=c("peace", "war"))

# distribute workload over multiple cores for faster computation 
registerDoMC(cores=7)
set.seed(666)

tc<-trainControl(method="cv", number=10, summaryFunction=twoClassSummary, classProb=T,
                 savePredictions = T)

#Fearon and Laitin Model (2003) Specification###
model.fl.1<- train(as.factor(warstds)~warhist+ln_gdpen+lpopns+lmtnest+ncontig+oil+nwstate +inst3+pol4+ef+relfrac, #FL 2003 model spec
                   metric="ROC", method="glm", family="binomial", trControl=tc, data=data.full)
summary(model.fl.1) 
model.fl.1

### Collier and Hoeffler (2004) Model specification###
model.ch.1<- train(as.factor(warstds)~sxpnew+sxpsq+ln_gdpen+gdpgrowth+warhist+lmtnest+ef+popdense
                   +lpopns+coldwar+seceduc+ptime,
                   metric="ROC", method="glm", family="binomial",
                   trControl=tc, data=data.full)
summary(model.ch.1)
model.ch.1

### Hegre and Sambanis (2006) Model Specification###
model.hs.1<- train(warstds~lpopns+ln_gdpen+inst3+parreg+geo34+proxregc+gdpgrowth+anoc+ partfree+nat_war+lmtnest+decade1+pol4sq+nwstate+regd4_alt+etdo4590+milper+ geo1+tnatwar+presi,
                   metric="ROC", method="glm", family="binomial",
                   trControl=tc, data=data.full)
summary(model.hs.1)
model.hs.1


### Random Forest ###
model.rf<-train(as.factor(warstds)~., metric="ROC", method="rf", sampsize=c(30,90),
                importance=T,proximity=F, ntree=1000, trControl=tc, data=data.full) 
summary(model.rf)
model.rf

### Full Logistic Model: Train logistic model on all 88 variables ###
model.full_logit.1 <- train(as.factor(warstds)~ager+agexp+anoc+army85+autch98+auto4+autonomy+avgnabo+centpol3+coldwar+decade1+decade2+decade3+decade4+dem+dem4+demch98+dlang+drel+durable+ef+ef2+ehet+elfo+elfo2+etdo4590+expgdp+exrec+fedpol3+fuelexp+gdpgrowth+geo1+geo2+geo34+geo57+geo69+geo8+illiteracy+incumb+infant+inst+inst3+life+lmtnest+ln_gdpen+lpopns+major+manuexp+milper+mirps0+mirps1+mirps2+mirps3+nat_war+ncontig+nmgdp+nmdp4_alt+numlang+nwstate+oil+p4mchg+parcomp+parreg+part+partfree+plural+plurrel+pol4+pol4m+pol4sq+polch98+polcomp+popdense+presi+pri+proxregc+ptime+reg+regd4_alt+relfrac+seceduc+second+semipol3+sip2+sxpnew+sxpsq+tnatwar+trade+warhist+xconst,
                      metric="ROC", method="glm", family="binomial", trControl=tc, data=data.full)
summary(model.full_logit.1)
model.full_logit.1

### Combined Model: Combine Fearon/Latin and Collier and Hoeffler 
model.combined_fl_ch.1<- train(as.factor(warstds)~warhist+ln_gdpen+lpopns+lmtnest+ncontig+oil+nwstate+inst3+pol4+ef+relfrac+sxpnew+sxpsq+gdpgrowth+warhist+popdense+coldwar+seceduc+ptime, 
                   metric="ROC", method="glm", family="binomial", trControl=tc, data=data.full)
summary(model.combined_fl_ch.1) 
model.combined_fl_ch.1

### Insurgency Model: Use features that might back up Fearon/Laitin's hypothesis that civil war occurs when conditions are beneficial to insurgents
model.insurgency.1<- train(as.factor(warstds)~warhist+ln_gdpen+illiteracy+sxpsq+seceduc+milper+mirps2+inst3+pol4+warhist+ncontig+lmtnest, 
                           metric="ROC", method="glm", family="binomial", trControl=tc, data=data.full)
summary(model.insurgency.1) 
model.insurgency.1

### Significant Features Model: Use features that are statistically significant with p < 0.05, according to the full logistic model
model.stat_sig.1<- train(as.factor(warstds)~decade2+ef2+gdpgrowth+inst3+life+nwstate+parreg+presi+proxregc+second+sxpnew+sxpsq, 
                           metric="ROC", method="glm", family="binomial", trControl=tc, data=data.full)
summary(model.stat_sig.1) 
model.stat_sig.1

### ROC Plots for Different Models ###
library(ROCR)
attach(data.full)

pred.FL.war<-model.fl.1$finalModel$fitted.values 
pred.CH.war<-model.ch.1$finalModel$fitted.values 
pred.HR.war<-model.hs.1$finalModel$fitted.values

RF.1.pred<-predict(model.rf$finalModel, type="prob") 
RF.1.pred<-as.data.frame(RF.1.pred)

pred.FULL_LOGIT.war<-model.full_logit.1$finalModel$fitted.values
pred.COMBINED_fl_ch.war<-model.combined_fl_ch.1$finalModel$fitted.values
pred.INSURGENCY.war<-model.insurgency.1$finalModel$fitted.values
pred.STAT_SIG.war<-model.stat_sig.1$finalModel$fitted.values

### Plot ROC Curves (Corrected)### (Originally shown on Figure 2 page 10)
pred.FL <- prediction(pred.FL.war, data.full$warstds)
perf.FL <- performance(pred.FL,"tpr","fpr")
pred.CH <- prediction(pred.CH.war, data.full$warstds)
perf.CH <- performance(pred.CH,"tpr","fpr") 
pred.HS<-prediction(pred.HR.war, data.full$warstds) 
perf.HS<-performance(pred.HS, "tpr", "fpr")

pred.RF.1<-prediction(RF.1.pred$war, data.full$warstds) 
perf.RF.1<-performance(pred.RF.1, "tpr", "fpr")

pred.FULL_LOGIT<-prediction(pred.FULL_LOGIT.war, data.full$warstds) 
perf.FULL_LOGIT<-performance(pred.FULL_LOGIT, "tpr", "fpr")
pred.COMBINED_fl_ch<-prediction(pred.COMBINED_fl_ch.war, data.full$warstds) 
perf.COMBINED_fl_ch<-performance(pred.COMBINED_fl_ch, "tpr", "fpr")
pred.INSURGENCY<-prediction(pred.INSURGENCY.war, data.full$warstds) 
perf.INSURGENCY<-performance(pred.INSURGENCY, "tpr", "fpr")
pred.STAT_SIG<-prediction(pred.STAT_SIG.war, data.full$warstds) 
perf.STAT_SIG<-performance(pred.STAT_SIG, "tpr", "fpr")

### Code for plotting the corrected ROC Curves in Figure 1.
pdf("Fig2_ExtendedLogit.pdf", 7, 7)
plot(perf.FL, main="Uncorrected Logits and Random Forests (Corrected)", col='black')
plot(perf.CH, add=T, lty=2, col='red')
plot(perf.HS, add=T, lty=3, col='blue')
plot(perf.RF.1, add=T, lty=4, col='green')

plot(perf.FULL_LOGIT, add=T, lty=5, col='orange')
plot(perf.COMBINED_fl_ch, add=T, lty=6, col='purple')
plot(perf.INSURGENCY, add=T, lty=7, col='pink')
plot(perf.STAT_SIG, add=T, lty=8, col='brown')

legend(0.5,0.5, legend=c("Fearon and Laitin (2003) 0.77", "Collier and Hoeffler (2004) 0.82", "Hegre and Sambanis (2006) 0.80", "Random Forest 0.91", "Full Logit 0.83", "Combined FL + CH 0.847", "Insurgency 0.80", "Statistically Significant 0.85"), 
       lty=c(1,2,3,4,5,6,7,8), col=c("black", "red", "blue", "green", "orange", "purple", "pink", "brown"), bty="n", cex = .75)
dev.off()

###Separation Plots### 
library(separationplot)

## Transform DV back to 0,1 values for separation plots. 
data.full$warstds<-factor(
  data.full$warstds,
  levels=c("peace","war"),
  labels=c(0, 1))

# Transform actual observations into vector for separation plots. 
Warstds<-as.vector(data.full$warstds)

### Corrected Separation Plots###

separationplot(RF.1.pred$war, Warstds, type = "line", line = T, lwd2=1, show.expected=T, heading="Random Forests", height=2.5, col0="white", col1="black")
separationplot(pred.FL.war, Warstds, type = "line", line = T, lwd2=1, show.expected=T, heading="Fearon and Laitin (2003)", height=2.5, col0="white", col1="black")
separationplot(pred.CH.war, Warstds, type = "line", line = T, lwd2=1, show.expected=T, heading="Collier and Hoeffler (2004)", height=2.5, col0="white", col1="black")
separationplot(pred.HR.war, Warstds, type = "line", line = T, lwd2=1, show.expected=T, heading="Hegre and Sambanis (2006)", height=2.5, col0="white", col1="black")
separationplot(pred.FULL_LOGIT.war, Warstds, type="line", line = T, lwd2=1, show.expected=T, heading="Full Logistic Regression", height=2.5, col0="white", col1="black")
separationplot(pred.COMBINED_fl_ch.war, Warstds, type="line", line = T, lwd2=1, show.expected=T, heading="Combined FL+CH", height=2.5, col0="white", col1="black")
separationplot(pred.INSURGENCY.war, Warstds, type="line", line = T, lwd2=1, show.expected=T, heading="Insurgency", height=2.5, col0="white", col1="black")
separationplot(pred.STAT_SIG.war, Warstds, type="line", line = T, lwd2=1, show.expected=T, heading="Statistically Significant", height=2.5, col0="white", col1="black")

### Data imputation ###
# Seed for Imputation of out-of-sample data. 
set.seed(425)

# Dataset for imputation. 
data_imp<-read.csv(file="../dataverse_files/data_full_fromonline.csv")

# Imputation procedure.
rf.imp<-rfImpute(data_imp, as.factor(data_imp$warstds), iter=5, ntree=1000)

### Out of Sample Data ###
# Subsetting imputed data. 
mena<-subset(rf.imp, rf.imp$year > 2000)

### Generate out of sample predictions for Table 1 (corrected)
fl.pred<-predict(model.fl.1, newdata=mena, type="prob") 
fl.pred<-as.data.frame(fl.pred)
pred.FL.1<-prediction(fl.pred$war, mena$`as.factor(data_imp$warstds)`) 
perf.FL.1<-performance(pred.FL.1, "auc")

ch.pred<-predict(model.ch.1, newdata=mena, type="prob")
ch.pred<-as.data.frame(ch.pred)
pred.CH.1<-prediction(ch.pred$war, mena$`as.factor(data_imp$warstds)`) 
perf.CH.1<-performance(pred.CH.1, "auc")

hs.pred<-predict(model.hs.1, newdata=mena, type="prob") 
hs.pred<-as.data.frame(hs.pred)
pred.HS.1<-prediction(hs.pred$war, mena$`as.factor(data_imp$warstds)`) 
perf.HS.1<-performance(pred.HS.1, "auc")

rf.pred<-predict(model.rf, newdata=mena, type="prob")
rf.pred<-as.data.frame(rf.pred)
pred.RF.1<-prediction(rf.pred$war, mena$`as.factor(data_imp$warstds)`)
perf.RF.1<-performance(pred.RF.1, "tpr", "fpr") 
perf.RF.1<-performance(pred.RF.1, "auc")

full_logit.pred<-predict(model.full_logit.1, newdata=mena, type="prob") 
full_logit.pred<-as.data.frame(full_logit.pred)
pred.FULL_LOGIT.1<-prediction(full_logit.pred$war, mena$`as.factor(data_imp$warstds)`) 
perf.FULL_LOGIT.1<-performance(pred.FULL_LOGIT.1, "auc")

combined_fl_ch.pred<-predict(model.combined_fl_ch.1, newdata=mena, type="prob") 
combined_fl_ch.pred<-as.data.frame(combined_fl_ch.pred)
pred.COMBINED_fl_ch.1<-prediction(combined_fl_ch.pred$war, mena$`as.factor(data_imp$warstds)`) 
perf.COMBINED_fl_ch.1<-performance(pred.COMBINED_fl_ch.1, "auc")

insurgency.pred<-predict(model.insurgency.1, newdata=mena, type="prob") 
insurgency.pred<-as.data.frame(insurgency.pred)
pred.INSURGENCY.1<-prediction(insurgency.pred$war, mena$`as.factor(data_imp$warstds)`) 
perf.INSURGENCY.1<-performance(pred.INSURGENCY.1, "auc")

stat_sig.pred<-predict(model.stat_sig.1, newdata=mena, type="prob") 
stat_sig.pred<-as.data.frame(stat_sig.pred)
pred.STAT-SIG.1<-prediction(STAT-SIG.pred$war, mena$`as.factor(data_imp$warstds)`) 
perf.STAT-SIG.1<-performance(pred.STAT-SIG.1, "auc")

### Save Imputed Data. ###
predictions<-cbind(mena$cowcode, mena$year, mena$warstds, fl.pred[,2], ch.pred[,2], hs.pred[,2], rf.pred[,2], full_logit.pred[,2], combined_fl_ch.pred[,2], insurgency.pred[,2], stat_sig.pred[,2])

### Write column headings for the out of sample data. ### 
colnames(predictions)<-c("COWcode", "year", "CW_Onset", "Fearon and Latin (2003)", "Collier and Hoeffler (2004)", "Hegre and Sambanis (2006)",
                         "Random Forest", "Full Logistic Regression", "Combined FL + CH", "Insurgency", "Statistically Significant")

### Save predictions as data frame for ordering the columns. 
predictions<-as.data.frame(predictions)

### Table 1 Results, ordered by Onset (decreasing), and year (increasing) in R rather than excel.
Onset_table<-predictions[order(-predictions$CW_Onset, predictions$year),]

### Rows 1-19 of the above go in Table 1. ### 
Onset_table_1thru19<-head(Onset_table, n=19)

### Here's the code for Table 1 in Latex. ### 
xtable(Onset_table_1thru19)

### Write the .csv file for all predictions to check against the Latex code for Table 1. 
### Sort the csv same way as the Latex table - CW_Onset (decreasing), then by year (increasing).
write.csv(predictions, file="extended-logit-tables/extendedLogisticRegression.csv")
