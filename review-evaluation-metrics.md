### Review of Evaluation Metrics

In this summary, I would like to list evaluation metrics of _Music Generation_ papers and their summaries.

1) **Distribution of Pitches**
   
   Idea comes from _Folk-RNN_ paper. They compare distribution of pitches of datasets and generated outputs. I think, we can easily implement this metric. _(This is **unary** feature.)_

2) **Distribution of Number of Tokens**
   
    Idea comes from _Folk-RNN_ paper. They compare number of tokens in a song (how many token we have until the token which represent end of the sequence) from dataset and generated outputs. In our case, for example, **-1** in _Koma53_ represent end of the sequence. Thus, we can easily implement this metric.

    ![Alt Text](https://docs.google.com/uc?id=1JhQYSYsLzZRtejPY3BvwpASiXohbyodw)

3) **Each section will end with a resolution**
   
    When researcher checks the output of the _Folk-Rnn_, they realize that each section will end with a resolution. We can use this type of **spesific** metric for our case. 

4) **Transition Matrix of Pitch and Duration**
   
    The idea comes from [Algorithmic Composition of Melodies with Deep Recurrent Neural Networks](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#4-algorithmic-composition-of-melodies-with-deep-recurrent-neural-networks), they compare the transition matrix of _pitch_ and _duration_. We can easily implement it and compare. _(This is **bi-gram** feature.)_

    ![Alt Text](https://docs.google.com/uc?id=1PzHbnqOvvSWcuZg91mdkr426R-MYlG2U)

5) **Conservation of Metric Structure**
   
    Idea comes from [Algorithmic Composition of Melodies with Deep Recurrent Neural Networks](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#4-algorithmic-composition-of-melodies-with-deep-recurrent-neural-networks), they use _Irish Music_ as dataset and realized that system learns that from a rhythmical point of view, it is interesting to notice that, even though the model had no notion of bars implemented, the metric structure was preserved in the generated continuations.

6) **Mutual Information with Time**
   
    I saw this idea at [Music Generation with Variational Recurrent Autoencoder Supported by History](https://github.com/hedonistrh/TurkishMusicGeneration/blob/master/2018-10-10-Literature-Review-for-Music-Generation.md#7-music-generation-with-variational-recurrent-autoencoder-supported-by-history). Main source of this metric is [Critical Behavior from Deep Dynamics: A Hidden Dimension in Natural Language](https://cbmm.mit.edu/sites/default/files/publications/1606.06737.pdf)

    * **Introduction of the paper:** _We show that in many data sequences — from texts in different languages to melodies and genomes — the mutual information between two symbols decays roughly like a power law with the number of symbols in between the two. In contrast, we prove that Markov hidden Markov processes generically exhibit exponential decay in their mutual information, which explains why natural languages are poorly approximated by Markov processes. We present a broad class of models that naturally reproduce this critical behavior._
    ![Alt Text](https://docs.google.com/uc?id=1WFFHMJvo38CIluiC8B4b7j42N7JUfcPC)

    * [This stackoverflow question](https://stats.stackexchange.com/questions/241432/calculating-mutual-information-over-distance) can be helpful.

    * Ps. Mutual information is a quantity that measures a relationship between two random variables that are sampled simultaneously. In particular, it measures how much information is communicated, on average, in one random variable about another

7) **Cross Entropy**
   
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

13) **Average pitch interval (PI)**
    
    Average value of the
    interval between two consecutive pitches in semitones.
    The output is a scalar for each sample.

14) **Note Count**
    
    The number of used notes. As
    opposed to the pitch count, the note count does not
    contain pitch information but is a rhythm-related
    feature. The output is a scalar for each sample.

15) **Average inter-onset-interval (IOI)**

    To calculate the
    inter-onset-interval in the symbolic music domain,
    we find the time between two consecutive notes. The
    output is a scalar in seconds for each sample.

16) **Pitch range (PR)**
    
    The pitch range is calculated by
    subtraction of the highest and lowest used pitch in
    semitones. The output is a scalar for each sample


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
  - For this one, we can use _T-SNE_ and _UMAP_ as unsupervised classification. For my graduation project, we have tried it.
    - **Method**: The first feature we came up with was the logarithm of frequency distance between decision note and every other note matrix for each piece. We accepted decision note of a piece as the very last note of each piece. After that, since each song included a variety of notes of different bemol and diez degrees, we created a dictionary for each variation of note in specific octave with specific bemol or diez degree. And to create the resulting matrix, we calculated the Euclidean metric distance between each note and decision note to form the draft of our first feature set. And afterward, we transformed the distance matrix for each song into a fixed sized distance matrix which was of the same format for each song. The resulting matrix was the first feature set of our feature matrix.
  
        As a second feature, we formed note frequency histograms for each song. And bins of these frequency histograms were the second feature set of our feature matrix.

        Lastly, we created a one-hot matrix for the type of the makam song. Each song in different makam was written with a different method and by using the information of the song, we extracted the method it was written in. Afterward, for each method, we used a different feature set to represent the song altogether with previous feature sets.

        After these transformations, we have applied our unsupervised methods for makam classification. We have also method and form, however, according to experts of Turkish Makam Music makam is the most important classifier for the emotion.

    - **Results**
    ![Alt Text](https://docs.google.com/uc?id=0B-6ztEhriyaAaFZ0UWFpTTRiRGZ5YnhhVFE1NzRsbU54V0FJ)
    ![Alt Text](https://docs.google.com/uc?id=0B-6ztEhriyaAdHVzRC1aeXpjVEhocFVmbFBycXNadzVBMnJn)


- Usul Classification
- User studies
- Note Distribution of the first section, second section etc.

-----------
TO-DO
- Read [On the evaluation of generative models in music](http://www.musicinformatics.gatech.edu/wp-content_nondefault/uploads/2018/11/postprint.pdf)
