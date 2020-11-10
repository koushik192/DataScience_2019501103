setwd("C:\\Users\\Chowdary\\Desktop\\Finalexam\\Finalexam")
bse <- read.csv("BSE_Sensex_Index.csv")
S_Close <- c()
for (i in 1:15446){
  S_Close[i] <- (bse$Close[i] - bse$Close[i+1]) / bse$Close[i+1]
}
S_Close[15447] <- (S_Close[15446] + S_Close[15445] + S_Close[15444]) / 3
S_Close[15447]
Z_S_Close <- c()
mean_S_Close <- mean(SGR_Close)
mean_S_Close
sd_S_Close <- sd(SGR_Close)
sd_S_Close
for (j in 1:15447) {
  Z_S_Close[j] <- (S_Close[j] - mean_S_Close) / (sd_S_Close) 
} 
o_d <- c()
o_c <- 0
otd <- 1
for (k in 1:15447) {
  if (Z_S_Close[k] > 3) {
    o_c <- o_c + 1
    o_d[otd] <- bse$Date[k]
    otd <- otd + 1
  }
  if (Z_S_Close[k] < -3) {
    o_c <- o_c + 1
    o_d[otd] <- bse$Date[k]
    otd <- otd + 1
  }
}
o_c
o_d