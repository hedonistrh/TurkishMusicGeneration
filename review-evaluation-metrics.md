In this summary, I would like to list evaluation metrics of _Music Generation_ papers and their summaries.

1) **Note Prediction**
   
   Idea comes from [Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#2-modeling-temporal-dependencies-in-high-dimensional-sequences-application-to-polyphonic-music-generation-and-transcription) 

2) **Distribution of Pitches**
   
   Idea comes from _Folk-RNN_ paper. They compare distribution of pitches of datasets and generated outputs. I think, we can easily implement this metric. _(This is **unary** feature.)_

3) **Distribution of Number of Tokens**
   
    Idea comes from _Folk-RNN_ paper. They compare number of tokens in a song (how many token we have until the token which represent end of the sequence) from dataset and generated outputs. In our case, for example, **-1** in _Koma53_ represent end of the sequence. Thus, we can easily implement this metric.

    ![Alt Text](https://docs.google.com/uc?id=1JhQYSYsLzZRtejPY3BvwpASiXohbyodw)

4) **Each section will end with a resolution**
   
    When researcher checks the output of the _Folk-Rnn_, they realize that each section will end with a resolution. We can use this type of **spesific** metric for our case. 

5) **Transition Matrix of Pitch and Duration**
   
    The idea comes from [Algorithmic Composition of Melodies with Deep Recurrent Neural Networks](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#4-algorithmic-composition-of-melodies-with-deep-recurrent-neural-networks), they compare the transition matrix of _pitch_ and _duration_. We can easily implement it and compare. _(This is **bi-gram** feature.)_

    ![Alt Text](https://docs.google.com/uc?id=1PzHbnqOvvSWcuZg91mdkr426R-MYlG2U)

6) **Conservation of Metric Structure**
   
    Idea comes from [Algorithmic Composition of Melodies with Deep Recurrent Neural Networks](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#4-algorithmic-composition-of-melodies-with-deep-recurrent-neural-networks), they use _Irish Music_ as dataset and realized that system learns that from a rhythmical point of view, it is interesting to notice that, even though the model had no notion of bars implemented, the metric structure was preserved in the generated continuations.

7) **Mutual Information with Time**
   
    I saw this idea at [Music Generation with Variational Recurrent Autoencoder Supported by History](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#7-music-generation-with-variational-recurrent-autoencoder-supported-by-history). Main source of this metric is [Critical Behavior from Deep Dynamics: A Hidden Dimension in Natural Language](https://cbmm.mit.edu/sites/default/files/publications/1606.06737.pdf)

    #### Add info about that

8) **Cross Entropy**
   
   Idea comes from [Music Generation with Variational Recurrent Autoencoder Supported by History](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#7-music-generation-with-variational-recurrent-autoencoder-supported-by-history) They compare _cross entropy of the architestures near saturation point_

   ![Alt Text](https://docs.google.com/uc?id=1JWEgNJKJrLCqmn_-R0Ob5tDeraAzs1hN)


Note that, 9-10-11 comes from [C-RNN-GAN](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#11-c-rnn-gan-continuous-recurrent-neural-networks-with-adversarial-training). Their implementation is available [in the repo.](https://github.com/olofmogren/c-rnn-gan)

9) **Scale Consistency**
    
    _Scale Consistency_ computed by counting the fraction of tones that were part of a standard scale, and reporting the number for the best matching such scale.

10)  **Tone span**

     _Tone span_ is the number of half-tone steps between the lowest and the highest tone in a sample.

11)  **Repetitions**
    
     _Repetitions of short subsequences_ were counted, giving a score on how much recurrence there is in a sample. This metric takes only the tones and their order into account, not their timing.

![Alt Text](https://docs.google.com/uc?id=1WxxzoatGb0byp0SBeg_eAjHtVyfMkJc9)

12)  **Qualified note rate(QN)**

        Idea comes from [CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS WITH BINARY NEURONS FOR POLYPHONIC MUSIC GENERATION](https://salu133445.github.io/bmusegan/pdf/bmusegan-ismir2018-paper.pdf) Qualified note rate (QN) computes the ratio of the number of the qualified notes (notes no shorter than three time steps, i.e., a 32th note) to the total number of notes. Low QN implies overly-fragmented music.

Now, lets look some metrics from [TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#14-tuning-recurrent-neural-networks-with-reinforcement-learning) These are based on music theory rules.

- **Notes Excessively Repeated**
- **Notes not in key**
- **Melodies starting with tonic**
- **Melodies with unique min and max note**
- **Notes in motif**
- **Notes in repeated motif**
- **Leaps  Resolved**
  
  ![Alt Text](https://docs.google.com/uc?id=1EQDE8U2MGJ-vCLikMie5RBDISsUmqVNL)




In our first meeting, we also discussed following metrics:
- Makam Classification
- Usul Classification
- User studies
- Note Distribution of the first section, second section etc.