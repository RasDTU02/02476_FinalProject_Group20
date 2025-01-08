# 02476_FinalProject_Group20

## Notes to us (To be deleted)
Reproducibility criterias:
  * Use Version Control
  * Use templates
    * Structuring of folders and files 
  * Write down your dependencies, and use virtual enviroments
    * Write a requirements.txt file
  * Document your code
    * Sections in README file documenting how to use and run
    * Sections TBD
  * Test your code
    * More on this to come 
  * Containerize your code
    * This is done with the Docker module
   
## Making the project
  * Initialize the data, and check it
    * Examine Summary Statistics (data normalized?)
    * Look at the distributions (is it shuffled?)
    * Visualize a few samples (are they corrupted?)
  * Start as simple as you can
    * Implement the most simple models in the beginning
    * Minimize the complexity
    * Then build from here so you understand what is going on
  * Make everything deterministic
    * Seed everythig and use the same seed everywhere
    * Remove all data augmentation
    * Use oly a single batch of data where you have a feeling of the outcome
  * Investigate the math
    * Go through the code, line by line
    * There should be a one-to-one match between equations and lines of codes
    * Refactor if not clear
    * Show shapes by using a comment line
    * Lookout for broadcasting
  * Overfitting is good?
    * Models should be able to memorize one batch
    * Train on one batch, if loss is 0 move on to larger models/larger data else debug
  * Really look at your loss
    * Are you printing the results correctly?
    * Remember the optimizer step with backpropagation
    * What about learning rate and batch size
  * Visualizations are your friend
    * Is training loss actually decreasing?
    * Can you add additional metrics?
    * Log dynamic changing parameters
    * Plot data, predictions, reconstructions etc. over time
  * Add complexity over time
    * Get a baseline that just works
    * Stop overfitting (regularization, data augmentations)
    * Tune hyper parameters
