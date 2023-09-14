library(mgcv) # just used to fit gam model
library(ggplot2)
library(gridExtra)
library(knitr)
library(kableExtra)
library(ggcorrplot)
library(ggpubr)
library(gamlss)
devtools::load_all()
options(knitr.kable.NA = '')
path <- "/Users/marquach/Desktop/github/deepregression_master-thesis_supple/"

options(orthogonalize = F)

# Fit some simple models like linear, additive and deep models with
# gam, deepregression tf and deepregression torch

# First a toy example with known true coefficients
x <- seq(from = 0, to = 1, 0.001)
beta1 <- 2
set.seed(42)
y <- rnorm(n = length(x), mean = 0*x, sd = exp(beta1*x))

toy_plot <- ggplot(data = toy_data, aes(x = x, y = y))+
  geom_point(shape = 1) +theme_classic()
ggsave(filename = paste0(path, "toy_plot.png"), width = 14, height = 8)

toy_gamlss <- gamlss(formula = y ~ 1+x,
       sigma.formula = ~ 1 + x, family = NO)

mod_torch <- deepregression(
  list_of_formulas = list(loc = ~ 1 + x, scale = ~ 1 + x),
  data = toy_data, y = toy_data$y,  engine = "torch")

mod_torch
# default lr = 0.001 change to 0.1 
#mod_torch$model <- mod_torch$model  %>% set_opt_hparams(lr = 0.01)
# torch approach does have a generic fit function 
mod_torch %>% fit(epochs = 500, early_stopping = F, validation_split = 0)

res_toy <- t(round(cbind(
  'gamlss' = c(toy_gamlss$mu.coefficients, toy_gamlss$sigma.coefficients),
  'torch' = c(rev(unlist(mod_torch %>% coef(which_param = 1))),
              rev(unlist(mod_torch %>% coef(which_param = 2)))),
  "true" = c(0, 0, 0, 2)), 3))
kable(res_toy, digits = 3, format = "latex") %>%
  add_header_above(header = c("$\\\\mu$" = 1:2, "$\\\\sigma^2$" = 3:4), escape = F)
#%>% 
kbl(res_toy, digits = 2, format = "latex", escape = F, booktabs = T,
    col.names = c("(Intercet)", "$\\beta$", "(Intercet)", "$\\beta$")) %>%
  add_header_above(header = c(" "=1, "$\\\\mu$" = 2, "$\\\\sigma^2$" = 2), escape = F) %>%
  writeLines(paste0(path,"toy_example-coefs.tex"))



################################################################################
################################################################################
#################### additive models #################### 
#### GAM data mcycles
data(mcycle, package = 'MASS')
plot(mcycle$times, mcycle$accel)
# Erst mit mgcv


gam_mgcv <- gam(mcycle$accel ~ 1 + s(times), data = mcycle)
gam_data <- model.matrix(gam_mgcv)
gam_torch <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel,
  from_preds_to_output = function(x, ...) x[[1]],
  loss = nnf_mse_loss,
  engine = "torch")

gam_tf <- deepregression(
  list_of_formulas = list(loc = ~ 1 + s(times), scale = ~ 1),
  data = mcycle, y = mcycle$accel,
  from_preds_to_output = function(x, ...) x[[1]],
  loss = "mse",
  engine = "tf")

# adapt learning rate to converge faster
gam_torch$model <- gam_torch$model  %>% set_opt_hparams(lr = 0.1)
gam_tf$model$optimizer$lr <- tf$Variable(0.1, name = "learning_rate")

gam_torch %>% fit(epochs = 2000, early_stopping = F, validation_split = 0)
gam_tf %>% fit(epochs = 2000, early_stopping = F, validation_split = 0)

# Interesting (does not converge when validation_loss is used as early_stopping)


mean((mcycle$accel - fitted(gam_mgcv) )^2)
mean((mcycle$accel - gam_torch %>% fitted(apply_fun = NULL))^2)
mean((mcycle$accel - gam_tf %>% fitted(apply_fun = NULL))^2)


plot_data <- rbind(data.frame(fitted = fitted(gam_mgcv),
                         Approach = "mgcv"),
                    data.frame(fitted = gam_torch %>% fitted(apply_fun = NULL),
                         Approach = "deepregression torch")
                   #data.frame(fitted = gam_tf %>% fitted(apply_fun = NULL),
                    #          Approach = "deepregression tf")
                   )
plot_data$times <- mcycle$times

rel_plot <- ggplot(data = mcycle, aes(x = times, y = accel))+
  geom_point(shape = 1)+
  ylab("Acceleration") + xlab("Time")+theme_classic()+
  geom_line(data = plot_data, aes(x = times, y = fitted, group = Approach,
                                  col = Approach))+
  theme(legend.position = c(0.8, 0.2),
        legend.background = element_rect(fill = "white"))

corr_data <- data.frame(
  #"tf" = gam_tf %>% fitted(apply_fun = NULL),
                        "torch" = gam_torch %>% fitted(apply_fun = NULL),
                        "mgcv" = fitted(gam_mgcv))

cor_plot <- ggplot(data = plot_data, aes(x = fitted, y = fitted, group = Approach))+
  geom_point(shape = 1)+
  ylab("Predicted Values (mgcv GAM)") + xlab("Predicted Values (deepregression torch)")+
  geom_abline(slope = 1)+
  theme_classic()

coef_comp <- rbind( 
  "mgcv GAM" = coef(gam_mgcv) ,
  "Deepreg Torch" = c(tail(unlist(gam_torch %>% coef()), 1),
              rev(tail(rev(unlist(gam_torch %>% coef())), -1))))



coef_tab <- ggtexttable(round(coef_comp,2), theme = ttheme("blank"))  %>%
  tab_add_hline(at.row = c(1, 2), row.side = "top", linewidth = 3, linetype = 1) %>%
  tab_add_hline(at.row = c(3), row.side = "bottom", linewidth = 3, linetype = 1) %>% 
  tab_add_title(text = "Bla bla", face = "bold", padding = unit(0.1, "line")) 

plot_mcycle <- ggarrange(rel_plot,cor_plot, ncol=2, nrow=1)
path <- "/Users/marquach/Desktop/github/deepregression_master-thesis_supple/"
ggsave(filename = paste0(path, "plot_mcycle.png"), width = 14, height = 8)
knitr::kable(coef_comp, format = "latex", digits = 2) %>% writeLines(paste0(path,"mcycle_coefs.tex"))



# load  and prepare data
airbnb <- readRDS("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/airbnb/munich_clean.RDS")
airbnb$days_since_last_review <- as.numeric(
  difftime(airbnb$date, airbnb$last_review)
)
y = log(airbnb$price)
orthog_options <- orthog_control(orthogonalize = F)
deep_model_tf <- function(x){
  x %>%
    layer_dense(units = 5, activation = "relu", use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 3, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "linear")
}

deep_model_torch <- function() nn_sequential(
  nn_linear(in_features = 2, out_features = 5, bias = F),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 5, out_features = 3, bias = T),
  nn_relu(),
  nn_dropout(p = 0.2),
  nn_linear(in_features = 3, out_features = 1))

mod_tf <- deepregression(y = y, data = airbnb,
                         list_of_formulas = 
                           list(
                             location = ~ 1 + beds +
                               s(accommodates, bs = "ps") +
                               s(days_since_last_review, bs = "tp") +
                               deep(review_scores_rating, reviews_per_month),
                             scale = ~1),
                         orthog_options = orthog_options,
                         list_of_deep_models = list(deep = deep_model_tf),
                         engine = "tf"
)
mod_torch <- deepregression(y = y, data = airbnb,
                            list_of_formulas = 
                              list(
                                location = ~ 1 + beds +
                                  s(accommodates, bs = "ps") +
                                  s(days_since_last_review, bs = "tp") +
                                  deep(review_scores_rating, reviews_per_month),
                                scale = ~1),
                            orthog_options = orthog_options,
                            list_of_deep_models = list(deep = deep_model_torch),
                            engine = "torch"
)

mod_tf %>% fit(epochs = 200, validation_split = 0.2, early_stopping = F)
mod_torch %>% fit(epochs = 200, validation_split = 0.2, early_stopping = F)

mu_tf_lin_coef <- unlist(rev(coef(mod_tf, type="linear")))
mu_torch_lin_coef <- unlist(rev(coef(mod_torch, type="linear")))

sigma_tf_lin_coef <- c(unlist(rev(coef(mod_tf, type="linear", which_param = 2))),
                       rep(NA, (length(mu_tf_lin_coef)-1)))
sigma_torch_lin_coef <- c(unlist(rev(coef(mod_torch, type="linear", which_param = 2))),
                          rep(NA, (length(mu_tf_lin_coef)-1)))
lin_coef <- data.frame(
  "tensorflow" = mu_tf_lin_coef,
  "torch" = mu_torch_lin_coef,
  "diff" = abs(mu_tf_lin_coef - mu_torch_lin_coef),
  "tensorflow" = sigma_tf_lin_coef,
  "torch" = sigma_torch_lin_coef,
  "diff" = abs(sigma_tf_lin_coef - sigma_torch_lin_coef))

lin_coef
rownames(lin_coef)[-1] <-  paste0("beds.",sort(unique(airbnb$beds)))[-1]
kbl(lin_coef, format = "latex", digits = 2, booktabs = T) %>% 
  add_header_above(header = c(" " = 1, "mu" = 3, "sigma" = 3)) %>%
  writeLines(paste0(path, "deep-lin_coefs.tex"))


pdf(file = paste0(path, "comp_tf-torch.pdf"), width = 14, height = 7)
par(mfrow = c(1,2))
which_plot <- 1
plot(mod_torch, which = which_plot, bty = "L", main = "")
par(new=TRUE)
plot(mod_tf, which = which_plot, bty = "L", axes = FALSE, pch = 16, main = "")
legend(x = 11,y = 0,
       legend = c("Torch", "Tensorflow"),
       pch = c(16, 1) )

which_plot <- 2
plot(mod_tf,  which = which_plot, main = "", bty = "L")
par(new=TRUE)
plot(mod_torch, which = which_plot, axes = FALSE, pch = 16, main = "")
legend(x = 2250,y = 0,
       legend = c("Torch", "Tensorflow"),
       pch = c(16, 1) )
dev.off()
par(mfrow = c(1,1))


fitted_vals_tf <- mod_tf %>% fitted()
fitted_vals_torch <- mod_torch %>% fitted()

panel.cor <- function(x, y, digits = 3, prefix = "", cex.cor, ...)
{
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y, method = "pearson"))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = 2.5)
}

pdf(file = paste0(path, "comp_fitted_tf-torch.pdf"), width = 14, height = 7)
pairs(data.frame(fitted_vals_tf, fitted_vals_torch, y),
      upper.panel = panel.cor, gap=0,
      labels = c("Fitted Tensorflow", "Fitted Torch", "Log-Price"))
dev.off()

# working with images
airbnb$image <- paste0("/Users/marquach/Desktop/R_Projects/semi-structured_distributional_regression/application/airbnb/data/pictures/32/",
                       airbnb$id, ".jpg")

cnn_block_tf <- function(
    filters,
    kernel_size,
    pool_size,
    rate,
    input_shape = NULL
){
  function(x){
    x %>%
      layer_conv_2d(filters, kernel_size,
                    padding="same", input_shape = input_shape) %>%
      layer_activation(activation = "relu") %>%
      layer_batch_normalization() %>%
      layer_max_pooling_2d(pool_size = pool_size) %>%
      layer_dropout(rate = rate)
  }
}

cnn_block_torch <- function(
    filters,
    kernel_size,
    pool_size,
    rate,
    shape = NULL
){
  function() nn_sequential(
    #layer_conv_2d(filters, kernel_size, padding="same", input_shape = input_shape)
    nn_conv2d(in_channels = 3, out_channels = filters,
              kernel_size = kernel_size, padding = "same"),
    nn_relu(),
    nn_batch_norm2d(num_features = filters),
    nn_max_pool2d(kernel_size = kernel_size),
    nn_dropout(p = rate)
  )
}
cnn_torch <- cnn_block_torch(
  filters = 16,
  kernel_size = c(3,3),
  pool_size = c(3,3),
  rate = 0.25,
  shape(200, 200, 3)
)

cnn_tf <- cnn_block_tf(
  filters = 16,
  kernel_size = c(3,3),
  pool_size = c(3,3),
  rate = 0.25,
  shape(200, 200, 3)
)

deep_model_cnn_tf <- function(x){
  x %>%
    cnn_tf() %>%
    layer_flatten() %>%
    layer_dense(32) %>%
    layer_activation(activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(1)
}

deep_model_cnn_torch <- function(){
  nn_sequential(
    cnn_torch(),
    nn_flatten(),
    nn_linear(in_features = 69696, out_features = 32),
    nn_relu(),
    nn_batch_norm1d(num_features = 32),
    nn_dropout(p = 0.5),  
    nn_linear(32, 1) )
}

mod_cnn_tf <- deepregression(
  y = y,
  list_of_formulas = list(
    ~1 + room_type + bedrooms + beds +
      deep_model_cnn(image),
    ~1 + room_type),
  data = airbnb,
  list_of_deep_models = list(deep_model_cnn =
                               list(deep_model_cnn_tf, c(200,200,3))))

mod_cnn_torch <- deepregression(
  y = y,
  list_of_formulas = list(
    ~1 + room_type + bedrooms + beds +
      deep_model_cnn(image),
    ~1 + room_type),
  data = airbnb,
  engine = "torch",
  orthog_options = orthog_options,
  list_of_deep_models = list(deep_model_cnn =
                               list(deep_model_cnn_torch, c(200,200,3))))

tictoc::tic()
mod_cnn_tf %>% fit(
  epochs = 100, batch_size = 32,
  early_stopping = F)
cnn_tf_time <- tictoc::toc()

tictoc::tic()
mod_cnn_torch %>% fit(
  epochs = 100, batch_size = 32,
  early_stopping = F)
cnn_torch_time <- tictoc::toc()



fitted_tf <- mod_cnn_tf %>% fitted()
fitted_torch <- mod_cnn_torch %>% fitted()

cnn_results <- data.frame("fitted_tf" = fitted_tf,
                          "fitted_torch" = fitted_torch)
cnn_results <- list("fitted" = cnn_results,
                    "time" = cbind(cnn_torch_time, cnn_tf_time))
save(cnn_results, file = paste0(path, "cnn_results.RData"))

plot(fitted_torch, fitted_tf)
cor(fitted_torch, fitted_tf)

cnn_coefs_mu <- data.frame("cnn_coef_tf" = unlist(coef(mod_cnn_tf, type = "linear")),
           "cnn_coef_torch" =  unlist(coef(mod_cnn_torch, type = "linear")))
cnn_coefs_mu <- rbind(cnn_coefs_mu[dim(cnn_coefs_mu)[1],],
                      cnn_coefs_mu[-dim(cnn_coefs_mu)[1],])
names_coef <- c("Intercept",
                paste("room type", levels(airbnb$room_type)[-1]),
                paste("bedrooms", levels(airbnb$bedrooms)[-1]),
                paste("beds", levels(airbnb$beds)[-1]))
                
rownames(cnn_coefs_mu) <- names_coef

cnn_coefs_sigma <- data.frame("cnn_coef_tf" = unlist(coef(mod_cnn_tf, type = "linear", which_param = 2)),
                           "cnn_coef_torch" =  unlist(coef(mod_cnn_torch, type = "linear", which_param = 2)))
cnn_coefs_sigma <- rbind(cnn_coefs_sigma[dim(cnn_coefs_sigma)[1],],
                         cnn_coefs_sigma[-dim(cnn_coefs_sigma)[1],])

cnn_coefs_sigma <- rbind(cnn_coefs_sigma,
                  data.frame( "cnn_coef_tf" = rep(NA, dim(cnn_coefs_mu)[1]-dim(cnn_coefs_sigma)[1]),
                              "cnn_coef_torch" = rep(NA, dim(cnn_coefs_mu)[1]-dim(cnn_coefs_sigma)[1])))
rownames(cnn_coefs_sigma) <- names_coef

cnn_coefs <- list("mu" = cnn_coefs_mu,
                  'sigma' = cnn_coefs_sigma)
save(cnn_coefs, file = paste0(path, "cnn_coefs.RData"))

cnn_coef_table <- data.frame(
  "tensorflow" = cnn_coefs$mu$cnn_coef_tf,
  "torch" = cnn_coefs$mu$cnn_coef_torch,
  "diff" = abs(cnn_coefs$mu$cnn_coef_tf - cnn_coefs$mu$cnn_coef_torch),
  "tensorflow" = cnn_coefs$sigma$cnn_coef_tf,
  "torch" = cnn_coefs$sigma$cnn_coef_torch,
  "diff" = abs(cnn_coefs$sigma$cnn_coef_tf - cnn_coefs$sigma$cnn_coef_torch))
rownames(cnn_coef_table) <- names_coef

kbl(cnn_coef_table, format = "latex", digits = 2, booktabs = T) %>% 
  add_header_above(header = c(" " = 1, "mu" = 3, "sigma" = 3)) %>%
  writeLines(paste0(path, "deep-ccn_coefs.tex"))

pdf(file = paste0(path, "multimodal_fitted_tf-torch.pdf"), width = 14, height = 7)
pairs(data.frame(fitted_tf, fitted_torch, y),
      upper.panel = panel.cor, gap=0,
      labels = c("Fitted Tensorflow", "Fitted Torch", "Log-Price"))
dev.off()

