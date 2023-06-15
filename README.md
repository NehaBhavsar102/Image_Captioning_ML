# Music-Genre-Classification

#Context
Music. For a very long time, experts have been attempting to comprehend sound and what makes one song different from another. ways to picture sound. what distinguishes one tone from another.

#About Dataset
Check out the dataset - https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Content
genres original - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
images original - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
2 CSV files - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.



#About Model
Used CNN Algorithm for our model's training. We chose this strategy because evidence from various types of research indicates that it produces the best outcomes for this issue.
Adam optimizer for training the CNN model. The 600th epoch was selected as the training model epoch.
The output layer uses the softmax function, while the output layer uses the RELU activation function for all hidden layers. Utilising the sparse_categorical_crossentropy function, the loss is calculated.
Overfitting is prevented by dropout.

After comparing various optimizers, we settled on the Adam optimizer because it produced the best results.
By increasing the number of epochs, the model's accuracy can be further improved, but after a while, we might reach a threshold, so the value should be chosen appropriately.


