if (FALSE==is.element("xts", installed.packages()[,1])) {
  install.packages("xts")
}

if (FALSE==is.element("zoo", installed.packages()[,1])) {
  install.packages("zoo")
}

if (FALSE==is.element("quantmod", installed.packages()[,1])) {
  install.packages("quantmod")
}

library(xts)
library(zoo)
library(quantmod)

btu <- getSymbols("BTU", src="yahoo")

print(btu)

barplot(BTU$BTU.Close)

print(BTU$BTU.Close)

prices <- BTU$BTU.Close

n = length(prices)

ManualRet <- diff(log(prices), lag=1)

plot(ManualRet)

AutoRet <- ROC(prices)

plot(AutoRet)

h = hist(AutoRet) 
h$density = h$counts/sum(h$counts)*100
plot(h,freq=FALSE)
lines(density(na.omit(AutoRet)))

nIter = 100000
fit_in = 0
ExpRet = mean(as.vector(AutoRet)[-1])
ExpVar = sqrt(var(as.vector(AutoRet)[-1]))

for(i in 1:nIter) {
   r = rnorm(5, mean=ExpRet, sd=ExpVar);
   if (sum(r) < -0.04) {
      fit_in = fit_in + 1  
   }
   
}
print(fit_in/nIter)
