
devtools::load_all()
save_res <- T
used_loss <- "distr_learning"
source("/Users/marquach/Desktop/github/deepregression_master-thesis_supple/speed_comparison/loop-fit-function.R")

set.seed(42)
n = c(500, 1000, 2000)              # 500, 1000, 2000
epochs <- c(100, 500, 1000) # 100, 500, 1000
sum(60*matrix(c(1, 2, 4, 5, 10, 20, 10, 20, 40), byrow = F, nrow = 3)*5)/60

nn_tf <- function(x) x %>%
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")


nn_torch <- nn_module(
  initialize = function(){
    self$fc1 <- nn_linear(in_features = 2, out_features = 32, bias = F)
    self$fc2 <-  nn_linear(in_features = 32, out_features = 1)
  },
  forward = function(x){
    x %>% self$fc1() %>% nnf_relu() %>% self$fc2()
  }
)

settings <- expand.grid(n = n, epochs = epochs)
setting_names <- lapply(1:nrow(settings),
                        function(x)
                          paste("n =",paste(settings[x,], collapse = ", epochs = ")
                          ))

formula <- ~ 1 + deep_model(x2,x3) + s(xa) + x1
orthog_options <- orthog_control(orthogonalize = F)

setting_res <- lapply(X = 1:nrow(settings), function(x){
  
  #setting combination
  n <- settings[x,1]
  epochs <- settings[x,2]
  
  # data generation
  data = data.frame(matrix(rnorm(4*n), c(n,4)))
  colnames(data) <- c("x1","x2","x3","xa")
  y <- rnorm(n) + data$xa^2 + data$x1
  
  semi_structured_torch <- deepregression(
    list_of_formulas = list(loc = formula, scale = ~ 1),
    list_of_deep_models = list(deep_model = nn_torch),
    data = data, y = y, orthog_options = orthog_options,
    engine = "torch")
  
  semi_structured_tf <- deepregression(
    list_of_formulas = list(loc = formula, scale = ~ 1),
    list_of_deep_models = list(deep_model = nn_tf),
    data = data, y = y, orthog_options = orthog_options,
    engine = "tf")
  
  # model for loop
  data_plain <- data[,-4]
  splines_data <- semi_structured_torch$init_params$parsed_formulas_contents$loc[[2]]$data_trafo()
  data_plain <- cbind(1, data_plain, splines_data)
  
  plain_torch_loop <- nn_module(
    initialize = function(){
      self$spline <- nn_linear(in_features = 9, out_features = 1, bias = F)
      self$linear <- nn_linear(in_features = 1, out_features = 1, bias = F)
      self$intercept_loc <- nn_linear(in_features = 1, out_features = 1, bias = F) 
      self$nn <- nn_torch()
      self$intercept_scale <- nn_linear(1, 1, F)
    },
    
    forward = function(input){
     loc <-  self$intercept_loc(input[,1, drop = F] ) + self$linear(input[,2, drop = F]) + self$nn(input[,3:4, drop = F]) +
        self$spline(input[,5:13, drop = F])
     scale <- torch_add(1e-8, torch_exp(self$intercept_scale(input[,1, drop = F])))
     distr_normal(loc = loc, scale = scale)
    }
  )
  
  model <- plain_torch_loop()
  
  # setup for luz
  # same as in deepregression (initialized outside)
  intercept_loc <- nn_linear(in_features = 1, out_features = 1, bias = F) 
  linear <- nn_linear(in_features = 1, out_features = 1, bias = F)
  spline <- nn_linear(in_features = 9, out_features = 1, bias = F)
  nn <- nn_torch()
  intercept_scale <- nn_linear(1, 1, F)
  
  plain_torch_luz <- nn_module(
    initialize = function(){
      self$spline <- spline
      self$linear <- linear
      self$intercept_loc <- intercept_loc
      self$nn <- nn
      self$intercept_scale <- intercept_scale
    },
    
    forward = function(input){
      loc <-  self$intercept_loc(input[,1, drop = F] ) + self$linear(input[,2, drop = F]) + self$nn(input[,3:4, drop = F]) +
        self$spline(input[,5:13, drop = F])
      scale <- torch_add(1e-8, torch_exp(self$intercept_scale(input[,1, drop = F])))
      distr_normal(loc = loc, scale = scale)
    }
  )
  plain_luz <- luz::setup(plain_torch_luz, 
                          loss = loss,
                          optimizer = optim_adam)

  tictoc::tic()
  benchmark_deepreg_semistruc <- bench::mark(
    "torch_loop" = plain_loop_fit_function(model, epochs = epochs, 
                                           batch_size = 32,
                                           data_x = data_plain, data_y = y,
                                           validation_split = 0.1, shuffle = T),
    "torch_luz" = plain_luz_fitted <- fit(plain_luz,
                                          data = list(as.matrix(data_plain), 
                                                      as.matrix(y)),
                                          epochs = epochs, verbose = F,
                                          valid_data = 0.1,
                                          dataloader_options = 
                                            list(batch_size = 32,
                                                 shuffle = T)),
    "deepregression_luz" = semi_structured_torch %>%
      fit(epochs = epochs,
          early_stopping = F,
          validation_split = 0.1,
          verbose = F, batch_size = 32), 
    "deepregression_tf" = semi_structured_tf %>% fit(epochs = epochs, 
                                                     early_stopping = F,
                                                     validation_split = 0.1, verbose = F,
                                                     batch_size = 32),
    memory = F, check = F, iterations = 5, relative = F)
  print(sprintf("%s: n=%s;epochs=%s setting \n took: %s",
               x, n, epochs, tictoc::toc()[4]))
  
  
  results <- list(
    "torch_loop" = rev(lapply(model$parameters, as.array)),
    "torch_luz" = lapply(plain_luz_fitted$model$parameters, as.array),
    "deepregression_luz" = rev(coef(semi_structured_torch)),
    "deepregression_tf" = rev(coef(semi_structured_tf,1)))

  benchmark_deepreg_semistruc$result <- results
  benchmark_deepreg_semistruc
  
})

names(setting_res) <- setting_names

if(save_res){
  save_name <- sprintf("/Users/marquach/Desktop/github/deepregression_master-thesis_supple/speed_comparison/results/speed_comparison_distr_%s.RData",
                       Sys.Date())
  save(setting_res, file = save_name)
}
load(file = "/Users/marquach/Desktop/github/deepregression_master-thesis_supple/speed_comparison/results/speed_comparison_distr_2023-08-18.RData")
setting_res_frame <- as.data.frame(
  Reduce(x = 
           lapply(setting_res, function(x) 
                              cbind(min = x[["min"]],
                                    max = sapply(x[["time"]], max),
                                    rel_max = as.numeric(sapply(x[["time"]], max)/min(sapply(x[["time"]], max))))),
  rbind))

setting_res_frame$n <- unlist(lapply(1:nrow(settings), function(x)
  rep(settings[x,1], 4))) # 4 different approaches
setting_res_frame$epochs <- unlist(lapply(1:nrow(settings), function(x)
  rep(settings[x,2], 4)))

approach <- c("torch_loop", "torch_luz", "deepregression_luz", "deepregression_tf")

setting_res_frame$approach <- approach

setting_res_frame$approach <- factor(setting_res_frame$approach, levels = 
                                       c("deepregression_tf",
                                         "torch_loop",
                                         "torch_luz",
                                         "deepregression_luz"),
                                     ordered = T)
levels(setting_res_frame$approach) <- c("Deepreg TF", "Torch-Loop", "Torch&Luz",
                                        "Deepreg Torch")
library(ggplot2)
path <- "/Users/marquach/Desktop/github/deepregression_master-thesis_supple/speed_comparison/results/"

benchmark_time <- ggplot(data = setting_res_frame,
                             aes(x = approach, y = max, fill = approach))+
  geom_col(position = "dodge2", show.legend = F,colour = "black")+
  geom_col(position = "dodge2", aes(y = min), show.legend = F, colour = "black",
           linetype = "dashed", alpha = 0)+
  facet_grid(n ~ epochs,
             labeller = labeller(
               epochs = c("100" = "epochs = 100",
                          "500" = "epochs = 500",
                          "1000" = "epochs = 1000"),
               n = c("500" = "n = 500",
                     "1000" = "n = 1000",
                     "2000" = "n = 2000")
             ), scales = "free_y")+
  theme_classic()+ ylab("Elapsed Time (s)") + xlab("Approaches")+
  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
  geom_text(data = setting_res_frame, aes(x = approach, y = max,
                                          label = round(max,2)),
            position = position_dodge(width = 1),
            vjust = -0.5, size = 3)+
  geom_text(data = setting_res_frame, aes(x = approach, y = 0.5*max,
                                          label = ifelse(
                                            round(rel_max,2)==1,
                                            yes = "", 
                                            no = round(rel_max,2))),
            position = position_dodge(width = 1),
            vjust = 0.5, size = 3)+
  scale_y_continuous(expand = c(0.05, 0, 0.2, 0))+
  scale_fill_grey()
benchmark_time
ggsave(filename = paste0(path,"distr_learning-time", ".png"), width = 14, height = 8)

benchmark_time_rel <- ggplot(data = setting_res_frame,
                             aes(x = approach, y = max))+
  geom_col(position = "dodge2", show.legend = F,colour = "black")+
  geom_col(position = "dodge2", aes(y = min), show.legend = F, colour = "black",
           linetype = "dashed", alpha = 0)+
  facet_grid(epochs ~ n,
             labeller = labeller(
               epochs = c("100" = "epochs = 100",
                          "500" = "epochs = 500",
                          "1000" = "epochs = 1000"),
               n = c("500" = "n = 500",
                     "1000" = "n = 1000",
                     "2000" = "n = 2000")
             ), scales = "free_y" 
             )+ # maybe add scales = "free_y"
  theme_classic()+ ylab("Elapsed Time (s)") + xlab("Approaches")+
  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
  geom_text(data = setting_res_frame, aes(x = approach, y = max,
                                              label = round(rel_max,2)),
            position = position_dodge(width = 1),
            vjust = -0.5, size = 3)+
  scale_y_continuous(expand = c(0.05, 0, 0.2, 0))
benchmark_time_rel
ggsave(filename = paste0(path,"distr_learning-time_rel", ".png"), width = 14, height = 8)


#benchmark_mem_abs <- ggplot(data = setting_res_frame, aes(x = test, y = mem_alloc, fill = test))+
#  geom_bar(stat = "identity", position = "dodge2")+
#  facet_grid(n ~ epochs,
#             labeller = labeller(
#               epochs = c("100" = "epochs = 100",
#                          "500" = "epochs = 500",
#                          "1000" = "epochs = 1000"),
#               n = c("500" = "n = 500",
#                     "1000" = "n = 1000",
#                     "2000" = "n = 2000")
#             ))+
#  theme_classic()+
#  ylab("Allocated Memory") + xlab("Approaches")+
#  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
#  geom_text(data = setting_res_frame, aes(x = test, y = as.numeric(mem_alloc),
#                                          label = round(mem_alloc,2)),
#            position = position_dodge(width = 1),
#            vjust = -0.5, size = 3)+
#  scale_y_continuous(expand = c(0.05, 0, 0.1, 0))
#benchmark_mem_abs

#benchmark_mem_rel <- ggplot(data = setting_res_frame, aes(x = test, y = mem_alloc, fill = test))+
#  geom_bar(stat = "identity", position = "dodge2")+
#  facet_grid(n ~ epochs,
#             labeller = labeller(
#               epochs = c("100" = "epochs = 100",
#                          "500" = "epochs = 500",
#                          "1000" = "epochs = 1000"),
#               n = c("500" = "n = 500",
#                     "1000" = "n = 1000",
#                     "2000" = "n = 2000")
#             ))+
#  theme_classic()+ ylab("Elapsed Time (s)") + xlab("Approaches")+
#  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
#  geom_text(data = setting_res_frame_rel, aes(x = test, y = setting_res_frame$mem_alloc,
#                                              label = round(mem_alloc,2)),
#            position = position_dodge(width = 1),
#            vjust = -0.5, size = 3)+
#  scale_y_continuous(expand = c(0.05, 0, 0.1, 0))
#benchmark_mem_rel


