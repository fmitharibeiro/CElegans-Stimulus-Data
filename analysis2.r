library(DataExplorer)

data <- read.csv("output/result.csv")

data$Smooth <- data$Smooth == "True"
data$Series <- as.factor(data$Series)
data$Start <- as.factor(data$Start)
data$Spikes <- as.factor(data$Spikes)
data$Sequence <- NULL

print(summary(data))

create_report(data)