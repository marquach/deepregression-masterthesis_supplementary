# low level loop used to train torch neural network in benchmark tests

loss_helper <- function(used_loss){
  switch (used_loss,
    "distr_learning" = function(input, target){
      torch_mean(-input$log_prob(target))
    },
    "mse" = function(input, target){
      nnf_mse_loss(input = input, target = target)
    }
  )
}

used_loss <- loss_helper(used_loss)

get_data <- torch::dataset(name = "test_loop",
                           initialize = function(x,y){
                             self$x <- x
                             self$y <- y
                           },
                           .getbatch = function(index){
                             x <- self$x[index,]
                             y <- self$y[index,]
                             list(x,torch_tensor(y))
                           },
                           .length = function(){
                             length(self$y)
                           }
                           )

device <- torch_device(if
                       (cuda_is_available()) {
                         "cuda"
                       } else {
                         "cpu" })

valid_batch <- function(b) {
  output <- model(b[[1]])
  target <- b[[2]]
  loss <- used_loss(output, target)
  loss$item()
}


train_batch <- function(b, optimizer) {
  optimizer$zero_grad()
  output <- model(b[[1]])
  target <- b[[2]]
  loss <- used_loss(output, target)
  loss$backward()
  optimizer$step()
  loss$item()
}

plain_loop_fit_function_dataloader <- function(model, epochs, batch_size, data_x, data_y,
                                    validation_split, verbose = F, shuffle = T){
  
  data_y <- as.matrix(data_y)
  num_data_points <- dim(data_y)[1]
  data_x <- torch_tensor(as.matrix(data_x))
  used_dataset <- get_data(x = data_x, y = data_y)
  
  # define index same like tensorflow
  train_ids <- 1:(ceiling((1-validation_split) * num_data_points))
  valid_ids <- setdiff(seq_len(num_data_points), train_ids)
  
  train_ds <- dataset_subset(dataset = used_dataset, indices = train_ids)
  valid_ds <-  dataset_subset(dataset = used_dataset, indices = valid_ids)
  
  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = shuffle)
  valid_dl <- dataloader(valid_ds, batch_size = batch_size, shuffle = shuffle)
  
  
  optimizer <- optim_adam(model$parameters)
  
  
  for (epoch in 1:epochs) {
    model$train()
    train_loss <- c()
    # use coro::loop() for stability and performance
    coro::loop(for (b in train_dl) {
      loss <- train_batch(b, optimizer)
      train_loss <- c(train_loss, loss)
    })
    cat(sprintf(
      "\nEpoch %d, training: loss: %3.5f \n",
      epoch, mean(train_loss)
    ))
    model$eval()
    valid_loss <- c()
  
    # disable gradient tracking to reduce memory usage
    with_no_grad({
      coro::loop(for (b in valid_dl) {
        loss <- valid_batch(b)
        valid_loss <- c(valid_loss, loss)
      }) })
    cat(sprintf(
      "\nEpoch %d, validation: loss: %3.5f \n",
      epoch, mean(valid_loss)
    )) }
  
}
