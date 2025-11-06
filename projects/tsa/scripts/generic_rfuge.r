### Import libraries
print("Unzip files")
untar("/opt/ml/processing/input/data/model.tar.gz", exdir="/opt/ml/processing/input/data/")

print("Import libraries")
# setwd("Rdeps_probscore")
print(list.files("/opt/ml/processing/input/data/"))
print(list.files("/opt/ml/processing/input/r_packages/"))
pkgFilenames <- read.csv("/opt/ml/processing/input/r_packages/pkgFilenames.csv", stringsAsFactors = FALSE)[, 1]

# First specify the packages of interest
packages <- c("splines", "quadprog", "glm2", "jsonlite", "dplyr")
to_install <- setdiff(packages, rownames(installed.packages()))
pkgNames <- unlist(lapply(to_install, FUN=function(x) {pkgFilenames[grep(x, pkgFilenames)]}))

# Now load or install&load all
# install.packages(pkgNames, repos = NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/glue_1.6.1.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/generics_0.1.2.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/rlang_1.0.1.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/lifecycle_1.0.1.tar.gz", repos=NULL, type="source")

install.packages("/opt/ml/processing/input/r_packages/magrittr_2.0.2.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/R6_2.5.1.tar.gz", repos=NULL, type="source")

install.packages("/opt/ml/processing/input/r_packages/ellipsis_0.3.2.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/fansi_1.0.2.tar.gz", repos=NULL, type="source")

install.packages("/opt/ml/processing/input/r_packages/cli_3.2.0.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/crayon_1.5.0.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/utf8_1.2.2.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/vctrs_0.3.8.tar.gz", repos=NULL, type="source")

install.packages("/opt/ml/processing/input/r_packages/pillar_1.7.0.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/pkgconfig_2.0.3.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/purrr_0.3.4.tar.gz", repos=NULL, type="source")

install.packages("/opt/ml/processing/input/r_packages/tibble_3.1.6.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/tidyselect_1.1.1.tar.gz", repos=NULL, type="source")

install.packages("/opt/ml/processing/input/r_packages/quadprog_1.5-8.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/glm2_1.2.1.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/jsonlite_1.7.3.tar.gz", repos=NULL, type="source")
install.packages("/opt/ml/processing/input/r_packages/dplyr_1.0.8.tar.gz", repos=NULL, type="source")

library(splines)
library(quadprog)
library(glm2)
library(jsonlite)
library(dplyr)

print("Libraries imported")

### Read score and tag files
print("Read score and tag files")
tag_file = '/opt/ml/processing/input/data/tag_file.csv'
score_file = '/opt/ml/processing/input/data/score_file.csv'

tag_data <- read.csv(tag_file,header=F)

score_data <- read.csv(score_file,header=F)

#set mininum records requirement to be 1000 and minimum number of fraud to be 10
min_record = 1000
min_fraud = 10

if(length(tag_data[[1]]) != length(score_data[[1]])) {
    status = "Fail"
    stop("The length of tags and score data are not equal. Check the input source.")
}

if(length(tag_data[[1]]) < min_record | sum(tag_data[[1]]) < min_fraud) {
    status = "Fail"
    stop("The data should contain too less records or too less positve records. Check the input source.")
}

if(!all(tag_data[[1]]>=0 & tag_data[[1]]<=1)){
    status = "Fail"
    stop("The tag are not in the range of 0 to 1. Check the input source.")
}

if(!all(score_data[[1]]>=0 & score_data[[1]]<=1)){
    status = "Fail"
    stop("The input model score are not in the range of 0 to 1. Check the input source.")
}

### Set hyperparameters
n_row = nrow(score_data)
degree = 3
# knots = 100
if(n_row > 1e6){
    knots = 100
}else if(n_row < 1e4){
    knots = 20
}else{
    knots = 50
}

input_knots <- knots
input_degree <- degree
input_additional_knots = ""

# Converting the additional knots input string data to the list of valid numeric values
additional_knots_string_vector = unlist(strsplit(input_additional_knots, ","))
additional_knots_string_vector_space_removed = gsub("\\s","",additional_knots_string_vector)
additional_knots_numeric=as.numeric(additional_knots_string_vector_space_removed)
additional_knots_with_invalid_numeric_removed = additional_knots_numeric[!is.na(additional_knots_numeric)&additional_knots_numeric>=0&additional_knots_numeric<=1]

### Genralized linear model with monotone BSplines
monotone_spline <- function(y,x,size,degree = 3,
                            boundary = c(0,1),knots=seq(0,1,length.out = 11),
                            tolerance = 1e-6,MaxIteration=1000,lambda=1e-10)
{
  nknots=length(knots)
  bs.nc <- nknots+degree-1

  bs.old <- bs(x,knots=knots[c(-1,-length(knots))],degree=degree,Boundary.knots = boundary,intercept=TRUE)
  tpr = y/size

  # Adding gruardrail to avoid a lot of memory consumption
  if (length(x) > 1000000) {
    ind = sample.int(length(x),1000000)
  } else {
    ind = 1:length(x)
  }

  glm_model = glm2(tpr[ind]~-1+bs.old[ind,], family="binomial", weights=size[ind])
  delta.old = glm_model$coefficients

  Dmat <- matrix(0,ncol=bs.nc,nrow=bs.nc-1)
  diag(Dmat) <- -1
  Dmat[cbind(1:(bs.nc-1),2:bs.nc)] <- 1

  #add the p-spline penalization matrix to:
  # 1. further control the smoothness
  # 2. increase numerical stability by avoiding ill matrix
  pspline_mat = matrix(0,ncol=bs.nc,nrow=bs.nc-2)
  diag(pspline_mat) = 1
  pspline_mat[cbind(1:(bs.nc-2),2:(bs.nc-1))] <- -2
  pspline_mat[cbind(1:(bs.nc-2),3:(bs.nc))] <- 1
  DD = crossprod(pspline_mat)

  for(j in 1:MaxIteration)
  {
    bs.eta <- bs.old%*%delta.old
    bs.mu <- 1/(1+exp(-bs.eta))
    bs.mu=ifelse(bs.mu==1, bs.mu-1e-9, bs.mu)
    bs.mu=ifelse(bs.mu==0, 1e-9, bs.mu)
    wt <- as.numeric(size*bs.mu*(1-bs.mu))
    z <- y-size*bs.mu

    # add a penalization on the smoothness to avoid ill matrix.
    #default lambda is 1e-10, small enough to not impact the curve but
    # able to avoid ill matrix.
    Pmat <- t(wt*bs.old)%*%(bs.old) + lambda*DD

    dvec <- t(bs.old)%*%(z+ (wt*bs.old)%*%delta.old)

    res.qp <- solve.QP(Dmat=Pmat,dvec=dvec,Amat=t(Dmat),bvec=rep(0,bs.nc-1))
    delta.update <- res.qp$solution

    diff.total<- sqrt(mean((delta.update - delta.old)^2))

    if(diff.total<= tolerance)
    { break}
    delta.old <- delta.update

  }

  fitted.values <- 1/(1+exp(-bs.old%*%delta.update))
  indicator = 1*(j==MaxIteration)

  res <- list(x=x, y=y, delta=delta.update,fitted.values = fitted.values,boundary=boundary,
              degree=degree,nknots=nknots,knots=knots,message=indicator)

  class(res) <- "MonotoneSpline"
  return(res)
}

### Run the monotone_spline method for the inputs and generate BSpline function Paramaters json
y=tag_data[[1]]
x=round(score_data[[1]],6)
size=rep(1, length(y))

# The following are to guarantee there are always some positive number of data points
# being the internal points of each interval;
# otherwise the bspline function may return NA.
#manually insert some quantiles close to 1 to make prob score closer to 1 on the right side
quantile_grid = sort(unique(c(seq(0, 1, length.out=input_knots+1),
                              seq(0.0, 1, length.out=10),
                              seq(0.5, 1, length.out=10),
                              seq(0.9, 1,length.out=10),seq(0.99, 1,length.out=10),
                              seq(0.999, 1, length.out=10), seq(0.9999, 1, length.out=10))))

if(!is.null(additional_knots_with_invalid_numeric_removed)&length(additional_knots_with_invalid_numeric_removed)>0){
    quantile_grid=sort(unique(c(quantile_grid, additional_knots_with_invalid_numeric_removed)))
}

x_quantiles_tmp = round(quantile(x, probs=quantile_grid), 8)
x_quantiles_tmp2 = unique(x_quantiles_tmp[-c(1, length(x_quantiles_tmp))]) #only keep the unique interior points
x_grid = seq(0,1,length.out=11)
x_quantiles = sort(unique(c(0, x_quantiles_tmp2, x_grid, 1)))

# count the number of data points in each interval split by the knots, excluding both the left and
#right ends of the interval or, for example, sum(0.1< x < 0.2) when knots are 0.1 and 0.2;
ncounts = hist(x[!(x %in% x_quantiles)], breaks=x_quantiles, plot=F)$counts
knots = c(0.0, x_quantiles[-1][ncounts>10]) # remove the right end of an interval if the number of data points <= 10;

if (max(knots) < 1) {
# reset the rightmost knot since 1 might be removed as well but 1 is always required as the last knot.
    #the following change change the end point of the last non empty interval to 1, so it won't be empty
  knots[length(knots)]=1.0
}

if (length(knots) <= 2) {
    print("The knots should at least contains one internal points besides 0 and 1; please check the distribution of the input")
    coefficients = NULL
    status = "Fail"
} else {
    df = data.frame(x=x, y=y)%>%
    group_by(x) %>%
    summarise(freq = n(), nbad = sum(y), .groups="keep")
    glm_model = monotone_spline(y=df$nbad, x=df$x, size=df$freq, knots=knots, degree=input_degree, boundary=c(0,1))
    coefficients=glm_model$delta
    tpr = df$nbad / df$freq
    #here mse is not weighted by d$freq to avoid being overweighted by any interval with a lot of points.
    model_mse = mean(sqrt((glm_model$fitted.values - tpr)^2))
    max_mse = mean(sqrt((mean(tpr) - tpr)^2))

    # set the max_coefficients to be 1e12 based on the failure case observed
    # set the min_unique_value to be 10 based on the failure case observed
    # this may not cover all failure cases
    max_coefficients = 1e12
    min_unique_value=10
    # Raise error if:
    # less than 10 uniques values generated in prediction
    # the model Mean squared error is larger than the maximum possible value
    # or some of the coefficients contains unusal large numbers
    if(sum(is.na(coefficients))==0 & length(unique(round(glm_model$fitted.values, 6))) > min_unique_value &
       model_mse < max_mse & max(abs(coefficients)) < max_coefficients){
        status = "Success"
    }else{
        status="Fail"
    }
}

### Create the bspline_parameters.json from result
#write down the key parameters
res=list(coefficients=coefficients, knots=c(rep(0, input_degree), knots, rep(1, input_degree)), degree=input_degree, status=status)
write_json(res, "/opt/ml/processing/output/bspline_parameters.json", digits = 8, auto_unbox=TRUE)
