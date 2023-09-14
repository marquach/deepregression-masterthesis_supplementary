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

loss <- loss_helper(used_loss)

plain_loop_fit_function <- function(model, epochs, batch_size, data_x, data_y,
                                    validation_split, verbose = F, shuffle = T){
  
  data_x <- torch_tensor(as.matrix(data_x))
  data_y <- torch_tensor(as.matrix(data_y))
  num_data_points <- data_y$size(1)
  
  # define index same like tensorflow
  train_ids <- 1:(ceiling((1-validation_split) * num_data_points))
  valid_ids <- setdiff(seq_len(num_data_points), train_ids)
  
  data_x_train <- data_x[train_ids,]
  data_x_valid <- data_x[valid_ids,]
  
  data_y_train <- data_y[train_ids]
  data_y_valid <- data_y[valid_ids]
  
  num_batches_train <- floor(data_y_train$size(1)/batch_size)
  num_batches_valid <- floor(data_y_valid$size(1)/batch_size)
  
  optimizer_t_manual <- optim_adam(model$parameters)
  
  for(epoch in 1:epochs){
    
    model$train()
    l_man <- c()
    
    # rearrange the data each epoch
    if(shuffle){
      permute <- torch_randperm(data_y_train$size(1)) + 1L
      data_x_train <- data_x_train[permute,]
      data_y_train <- data_y_train[permute]
    }
    
    # manually loop through the batches
    for(batch_idx in 1:num_batches_train){
      
      # here index is a vector of the indices in the batch
      index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
      
      x <- data_x_train[index,]
      y <- data_y_train[index]
      
      optimizer_t_manual$zero_grad()
      
      output <- model(x)
      loss_man <- loss(input = output, target = y)
      loss_man$backward()
      optimizer_t_manual$step()
      
      
      l_man <- c(l_man, loss_man$item())
    }
    if(verbose) cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l_man)))
    
    model$eval()
    valid_loss <- c()
    
    with_no_grad({
      # manually loop through the batches
      for(batch_idx in 1:num_batches_valid){
        
        # here index is a vector of the indices in the batch
        index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
        
        x <- data_x_valid[index,]
        y <- data_y_valid[index]
        
        output <- model(x)
        loss_val <- loss(input = output, target = y)
        
        valid_loss <- c(valid_loss, loss_val$item())
      }
      if(verbose) cat(sprintf("Valid loss at epoch %d: %3f\n", epoch, mean(valid_loss)))
      
    })
    
    
  }
}