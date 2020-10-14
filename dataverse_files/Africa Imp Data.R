setwd("Desktop")
getwd()
data<-read.csv(file="Africa_dat.csv")
attach(data)
library(randomForest)
newdata<-rfImpute(data, as.factor(warstds), iter=5, ntree=1000)

summary(newdata)

write.csv(newdata, file="AfricaImp.csv")
