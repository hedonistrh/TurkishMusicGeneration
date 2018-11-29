# Literature Review for Music Generation

There is a [excellent survey](https://hal.sorbonne-universite.fr/hal-01660772/document) for Music Generation, however, I would like to create up-to-date blogpost for who wants to start with old methods and track new papers. Also you can check this [excellent slide-set.](http://www-desir.lip6.fr/~briot/documents/deep-learning-music-generation-tutorial-ismir-2017.pdf) 

In the survey, researchers defines whole attempts via 4 main dimensions. 

- Objective
    - What type of musical content will be created. It can be melody, chord etc.

- Representation
    - The type of the data used to train and to generate musical content. It can be midi, piano roll, fourier transform of signal, ABC notation, different text notations etc.

- Architecture
    - What types of deep learning architecutes are used. It can be single architeture like [folk-rnn](https://folkrnn.org) or combination of different architecture like [VRASH](https://arxiv.org/pdf/1705.05458.pdf) Mostly, researcher use recurrent neural networks (RNN), convolutional neural network (CNN) and auto-encoder.

- Strategy
    - How architectures(systems) creates the musical content. It can be interpreted as prediction which is direct usage of the system. But, mostly researcher use indirect methods such as sampling, input manipulation.

I will skip the most of the explanation of these dimensions. Please refer [the survey](https://hal.sorbonne-universite.fr/hal-01660772/document) for more information.

Before the LSTM, there were some attempt for automatic music generation, however, their representation was designed with rich handcrafted
features and/or their results were not so satisfying. With the DL, these features are extracted by architectures. So that, I will directly start with LSTM paper.

##### To understand how RNN works for generation, please watch [Two Minute Papers video.](https://www.youtube.com/watch?v=Jkkjy7dVdaY)

If you want to check out papers which are before [Douglas Eck's LSTM](http://people.idsia.ch/~juergen/blues/IDSIA-07-02.pdf) paper:
- [Neural network music composition by prediction: Exploring the benefits of psychoacoustic constraints and multiscale processing](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.1924&rep=rep1&type=pdf)
- [A Connectionist Approach to Algorithmic Composition](https://pdfs.semanticscholar.org/81e0/a57abde1bf2cc4b7cd772e0573e92069e8ef.pdf)
    - "_Using RNNs for algorithmic composition allows to overcome the limitations of Markov chains in learning the long-range temporal dependencies of music."_
- [Connectionist models for real-time control of synthesis and compositional algorithms](http://cnmat.berkeley.edu/sites/default/files/attachments/1992_Connectionist-Models-for-Real-Time-Control-of-Synthesis.pdf)

### 1) [A First Look at Music Composition using LSTM Recurrent Neural Networks](http://people.idsia.ch/~juergen/blues/IDSIA-07-02.pdf)

##### Note: [My previous project](https://hedonistrh.github.io/2018-04-27-Music-Generation-with-LSTM/) is based on this paper. You can check that. :)

Traditional neural networks can not remember past information. They can only process current information. As you can think, if you can not remember past information, probably you can not even produce meaningful sentences. Recurrent Neural Network(RNN) solve this problem thanks to recurrent connection via loops at nodes. However, Vanilla RNN has another problem called as vanishing gradient. Before the LSTM, researchers use different RNN architecture to generate music, however, their models have failed to capture global musical structure. With the LSTM, researcher solved this problem. So that, they can capture the local and global structure of the music.

##### To understand how LSTM works, please check [Colah's excellent post.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

_The most straight-forward way to compose music with an RNN is to use the network as single-step predictor. The network learns to predict notes at time t + 1 using notes at time t as inputs. After learning has been stopped the network can be seeded with initial input values—perhaps from training data–and can then generate novel compositions by using its own outputs to generate subsequent inputs._

![Alt Text](https://s1.gifyu.com/images/try-2018-04-15-03.42.07.gif)

At this paper, they use one-hot encoding piano roll to represent data. When the note on they use 1.0, when the note off they use 0.0 to represent those steps. However, this represenation has some limitations. For instance, eight eighth notes of the same pitch are represented
exactly the same way as, say, four quarter notes of the same pitch. 

In simulations for this study, a range of 12 notes were possible for chords and 13 notes were possible for melodies. Training data is based on 12-bar blues. Each bar can contain 8 note. So that, a single song has 96 time steps. _(This type of representation is so limited.)_

They created 2 different experiment to understand can LSTM capture global and local structure of the music.

**Experiment 1: Learning Chords**

_In this experiment we show that LSTM can learn to reproduce a musical chord structure. My motivation is to ensure that the chord structure of the song can in fact be induced in absence of its melody. Otherwise it is unclear whether LSTM is taking advantage of the local structure in melody to predict global chord structure._

**Experiment 2: Learning Chords and Melody**

_The goal of the study was to see if LSTM could learn chord structure and melody structure and then use that structure to advantage when composing new songs_

Note that melody information does not reach nodes which are responsible from chords.


### 2) [Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf)

##### This project is open source. You can check their [source code](https://github.com/boulanni/theano-hf) and [tutorial.](http://deeplearning.net/tutorial/rnnrbm.html) Also you can check [thesis of Boulanger-Lewandowski](http://www-etud.iro.umontreal.ca/~boulanni/NicolasBoulangerLewandowski_thesis.pdf) for more information.

They use piano roll representation by 88 absolute pitch as representation. Their architecture is a combination of recurrent neural network(RNN) and restricted Boltzmann machines(RBM). At this paper, they have both generation and transcription. Thanks to transcription, they can easily compare their method via different methods such as MLP, RNN, note N-Gram, HMM-GMM etc.

_"Recurrent neural networks (RNN) incorporate an internal memory that can, in principle, summarize the entire sequence history. This property makes them well suited to represent long-term dependencies."_

For the polyphonic music, we have simultaneities notes and it occurs within correlated patterns. But RNN can not capture the nature of simultaneities note occuring. Because, RNN has been designed for multi class classification and it picks most probable one. Also, if we want to enumarate each of the probability, it can be so expensive. So that, we can use some energy-based models like RBM.

_"We wish to exploit the ability of RBMs to represent a complicated distribution for each time step, with parameters that depend on the previous ones."_

![Alt Text](http://deeplearning.net/tutorial/_images/rnnrbm.png)

They have 2 different experiment to understand performance of the proposed architecture.

- Note prediction.

![Alt Text](https://docs.google.com/uc?id=1kLVGKidgXtG3TqtKp0r1tTBpPJqDOzr2)

- Polyphonic Transcription

Multiple fundamental frequency (f0) estimation, or polyphonic transcription, consists in estimating the audible note pitches in the signal at 10 ms intervals without tracking note contours.

![Alt Text](https://docs.google.com/uc?id=1CWoS2ZztDlTGirUfi21IgMQlRzKr_Qvv)


### 3) [Music transcription modelling and composition using deep learning](https://arxiv.org/pdf/1604.08723.pdf)

##### This project is open source. You can check their [source code.](https://github.com/IraKorshunova/folk-rnn)

##### I also suggest that [Luke Johnston paper's](https://drive.google.com/file/d/0B7oYxDkqYqqPYlR2LU01LVktbUU/view) and [excellent YouTube video.](https://www.youtube.com/watch?v=aSr8_QQYpYM) His method is so similar with this paper.

Bob Sturm's and his team use LSTM to generate musical content. Their main difference is that they use text based representation. ([Douglas Eck's LSTM paper](http://people.idsia.ch/~juergen/blues/IDSIA-07-02.pdf) use piano roll representation.) 

They use [ABC notation.](http://abcnotation.com/wiki/abc:standard:v2.1) An entry begins with two identifiers, followed by the title, tune type, meter, key, ABC code, date, and contributing user.

    3038,3038,"A Cup Of Tea","reel","4/4","Amixolydian","|:eA (3AAA g2 fg |
    eA (3AAA BGGf|eA (3AAA g2 fg|1afge d2 gf:|2afge d2 cd||
    |:eaag efgf|eaag edBd|eaag efge|afge dgfg:|","2003-08-28 21:31:44","dafydd"
    3038,21045,"A Cup Of Tea","reel","4/4","Adorian","eAAa ~g2fg|eA~A2 BGBd|
    eA~A2 ~g2fg|1af (3gfe dG~G2:|2af (3gfe d2^cd||eaag efgf|
    eaag ed (3Bcd|eaag efgb|af (3gfe d2^cd:|","2013-02-24 13:45:39",
    "sebastian the megafrog"

They propose two different method.

- char-rnn: Operates over a vocabulary of single characters, and is trained on a continuous text file.
    - _"We keep only five ABC fields (title, meter, key, unit note length, and transcription), and separate each contribution by a blank line."_
    - There are 135 unique characters for char-rnn.

- folk-rnn: Operates over a vocabulary of transcription tokens, and is trained on single complete transcriptions.
    - They applied some pre-processing to transcriptions to get rid of high number of tokens like transpose. 
    - _"Each token consists of one or more characters — for the following seven types (with examples in parens): meter (“M:3/4”), key (“K:Cmaj”), measure (“:|” and “|1”), pitch (“C” and “^c’”), grouping (“(3”), duration (“2” and “/2”), and transcription (“<s” and “<\s>”."_ 
    - There are 137 unique tokens for folk-rnn.

_"Through training, char-rnn model learns a “language model” to produce ABC characters. On the contrary, folk-rnn model learns a language model in a vocabulary more specific to transcription, i.e., a valid transcription begins with <s, then a time signature token, a key token, and then a sequence of tokens from 4 types."_

We can look their compositions' statistic to understand that can the system capture the model of music.

- Distribution of pitches agree with the training data.
- Distribution of the number of tokens in a transcription agree with the training data.

![Alt Text](https://docs.google.com/uc?id=1JhQYSYsLzZRtejPY3BvwpASiXohbyodw)

_"The folk-rnn system seems to have learned about ending transcriptions on the tonic; and using measure tokens to create transcriptions with an AABB structure with each section being 8 measures long. In our latest experiments, we trained a folk-rnn system with transcriptions spelling out repeated measures (replacing each repeat sign with the repeated material). We find that many of the generated transcriptions  adhere closely to the AABB form, suggesting that this system is learning about repetition rather than where the repeat tokens occur."_

To understand the plausbility of the generated output, Bob Sturm and his team contacted with musicians. According to their response, some of the output is really plausible, however, some of them is not. 

### 4) [Algorithmic Composition of Melodies with Deep Recurrent Neural Networks](https://infoscience.epfl.ch/record/221014/files/AlgorithmicCompositionOfMelodiesWithDeepRecurrentNeuralNetworks.pdf) 

##### For RNN based methods, I also suggest that check other works of [Florian Colombo's.](https://infoscience.epfl.ch/search?f1=author&as=1&sf=title&so=a&rm=&m1=e&p1=Colombo%2C%20Florian&ln=en)

They represent music as a combination of notes which consist of pitch value and duration. They use one-hot encoding to represent pitch and duration.

![Alt Text](https://docs.google.com/uc?id=142I-Cr_qKbtyeGRvdC9FFx1zZSav0vdj)

As an architecture, they use two separate multi-layer RNNs. 

- _rhythm_ network: Takes durations of notes as an input and gives duration of the next note as an output.

-  _melody_ network: Takes pitch value and duration of next note and gives pitch value of the next note as an output
    
    **With the combination of those outputs, we have new note.**

![Alt Text](https://docs.google.com/uc?id=10lincOkCrEmgBhOZilr21np3qDI_jdZ3)

As a dataset, they use [2158 Irish melodies.](http://abc.sourceforge.net/NMD/)

### 5) [DeepHear - Composing and harmonizing music with neural networks](https://fephsun.github.io/2015/09/01/neural-music.html)

##### I highly recommend this blogpost. Also, you can check [source code.](https://github.com/fephsun/neuralnetmusic)

This method is based on autoencoder. Lets look the [Francois Chollet's explanation](https://blog.keras.io/building-autoencoders-in-keras.html) for autoencoder. 

_"Autoencoding" is a data compression algorithm where the compression and decompression functions are 1) data-specific, 2) lossy, and 3) learned automatically from examples rather than engineered by a human. Additionally, in almost all contexts where the term "autoencoder" is used, the compression and decompression functions are implemented with neural networks._

![Alt Text](https://blog.keras.io/img/ae/autoencoder_schema.jpg)

##### Also, you can read [this blogpost](https://hackernoon.com/how-to-autoencode-your-pokémon-6b0f5c7b7d97) to understand autoencoder via fancy way.

The main idea of the paper can be summarized as if we can reconstruct the original input via bottleneck layers, we feed that bottleneck layer with random input to create different output which can be interpreted as music. For the generation part, we just use decompressing. 

![Alt Text](https://fephsun.github.io/music/autoencoder-net.png)

Note that, they did not train directly whole system together. 

![Alt Text](https://fephsun.github.io/music/rbm-layers.png)

At first step, they just train the system with one bottleneck layer via restricted Boltzman machines.(RBM) At each step, they add one pair of layers to the network. (One for compressing part and one  for decompressing part) The old layers are fixed and just new layers' weights are uptated. At the final step, they train whole system jointly to fine-tune the system.

Pre-training an autoencoder one layer at a time, using restricted Boltzmann machines. At each step, a new pair of layers are added to the network: a compressing layer plus a corresponding decompressing layer. The weights of the new layers are then set to minimize the reconstruction error.

For representation, he uses one-hot encoding piano roll method. The music was converted into a binary matrix, where each column represents a 16-th note worth of time, and each row represents a pitch. At the each generation, it creates 4 bar sample and we have 80 different notes. So that input and output node has approximately 5000 node. (80 different note x 4 bar x 16 note/bar) Reconstruction creates some value between 0 and 1 according to activation function. With the threshold value, some of them become 1 and other ones become 0. 1 represent note on and 0 represent note off.

![Alt Text](https://fephsun.github.io/music/gen-net.png)

_Note: Before this project, [Andy Sarroff](https://andysarroff.com) and his team used similar method to create music, however, their input representation is constant Q transform. You can check their [source code](https://github.com/woodshop/deepAutoController) and [paper.](https://andysarroff.com/papers/sarroff2014a.pdf)_

The another benefit of the DeepHear is that you can manipulate the output via gradients of similarity. 

### 6) [VARIATIONAL RECURRENT AUTO-ENCODERS](https://arxiv.org/pdf/1412.6581.pdf)

Especially, this paper propose new type of Auto-Encoder and show the performance of the model via _music generation_. VRAE combines the strengths of Recurrent Neural Network(RNN) and Stochastic Gradient Variational Bayes(SVGB).

- Recurrent Neural Networks (RNNs) exhibit dynamic temporal behaviour which makes them suitable for capturing time dependencies in temporal data. (As opposed to simple auto-encoder, VRAE's encoder and decoder consist of recurrent layer.)

- Autoencoder maps the data to latent variables via encoder part. The Variational Bayesian approach maps the data to a distribution over latent variables. This type of network can be efficiently trained with Stochastic Gradient Variational Bayes (SGVB) which is a way to train models where it is assumed that the data is generated using some unobserved continuous random variable z.

_Given a latent space vector, the decoding part of the trained models can be used for generating data._

Let's look figures to understand that can latent space capture the musical content. 

![Alt Text](https://docs.google.com/uc?id=1Wl-BS4G2oZD0aVftQiBHB_8SNw5U38i-)
##### If latent space is 2-dimensional.

![Alt Text](https://docs.google.com/uc?id=14GfCJmYbfX0bAeaJMKRq9O_cUOfHl7ur)
##### If latent space is 20-dimensional. Representation has been done by T-SNE.

For the dataset, they use 8 MIDI file from 80s and 90s games. One-hot encoding piano roll representation has been used, however, they just include the most popular 49 notes from 88 notes. We can listen [the output of the architecture.](http://youtu.be/cu1_uJ9qkHA)

##### You can check [this blogspot](https://lirnli.wordpress.com/2017/09/27/variational-recurrent-neural-network-vrnn-with-pytorch/) for text generation via VRAE.

### 7) [Music Generation with Variational Recurrent Autoencoder Supported by History](https://arxiv.org/pdf/1705.05458.pdf)

##### You can check the researcher's [blogpost.](https://medium.com/altsoph/pianola-network-a069b95b6f39)

For the dataset, they use 15+ thousand normalized tracks. For the normalization, they applied different techniques like 
- The median pitch of every track was transposed to the 4th octave.
- Tracks with exceedingly small entropy were also excluded from the training data.

For the every note, they create concatenated note embedding which include pitch of the note, its octave, length of the note and meta-information.

For the baseline, they created language model.

![Alt Text](https://docs.google.com/uc?id=1f2_GL7I7vUkYBFnbm0rh1iJWxCXHg7iq)

After that, they created VAE with the same input. 

![Alt Text](https://docs.google.com/uc?id=1Ag0Cv3Cr0qqY14H0LEZ6-XN_wFwcDlv_)

They proposed VRASH(Variational Recurrent Autoencoder Supported by History). The difference with VAE, VRASH use previous outputs as an additional input. This is so similar with [VRAE](https://arxiv.org/pdf/1412.6581.pdf)) which is the previous paper. The difference is that VRASH use different representation for input. They seperately encode different informations and create note embedding.

![Alt Text](https://docs.google.com/uc?id=1jFREK5D-IxhWGfVSVgLjt8VyJ2hcnFEH)

For the comparison:

- They use cross-entropy.

![Alt Text](https://docs.google.com/uc?id=1JWEgNJKJrLCqmn_-R0Ob5tDeraAzs1hN)

- Compare the mutual information. 

![Alt Text](https://docs.google.com/uc?id=1prwe-297AYd9HrdRDhypDJzkgsE4Gc-g)


Generative models have some problems. 
- First general problem is that generative models have tendecy to repeat same notes. VRASH and VAE performs better than LM for this problem.
 
- Second general problem is about the macro structure of the generated music. Despite the fact that VAE and VRASH specifically are developed to capture macrostructures of the track they do not always provide distinct structural dynamics that characterizes a number of humanwritten musical tracks. However, VRASH seems to be the right way to go.


### 8) [DeepBach: a Steerable Model for Bach Chorales Generation](https://arxiv.org/pdf/1612.01010.pdf)

##### This paper is open source. Please check the [source code.](https://github.com/Ghadjeres/DeepBach)

_"We claim that, after being trained on the chorale harmonizations by Johann Sebastian Bach, our model is capable of generating highly convincing chorales in the style of Bach. DeepBach’s strength comes from the use of pseudo-Gibbs sampling coupled with an adapted representation of musical data."_

_"A key feature is that we are able to constrain the generated chorales in many ways: we can for instance impose the melody, the bass, the rhythm but also the cadences (when the musical phrases end)."_

To represent the data

- **Notes and Voices:** MIDI pitches to encode notes, discretize time with sixteenth notes.

- **Rhythm:** We choose to model rhythm by simply adding a hold symbol “__” coding whether or not the preceding note is held to the list of existing notes.

- **Metadata:** Normally, the music sheets contains more information like beat index, implicit metronome etc. For the DeepBach, researchers take into account the fermata symbol and current key signature. 

- **Chorale:** They represent the chorale as a combination of voices and metadata. 

![Alt Text](https://docs.google.com/uc?id=1Cs_XUtQ3t3Mc_HzRctD7GUSNUw9Qnmsm)

Where the aim is to predict a note knowing the value of its neighboring notes, the subdivision of the beat it is on and the presence of fermatas.

![Alt Text](https://docs.google.com/uc?id=1Syry5HiCCKEhmx-qU1D_e0B-n0US4g7K)

##### The first 4 lines represent voices, the bottom 2 lines represent metadata. This representation is just for 1 voice. For the 4 voice, this architecture is replicated 4 times. 

Aim is to predict a note knowing the value of its neighboring notes, the subdivision of the beat it is on and the presence of fermatas. The advantage with this formulation is that each classifier has to make predictions within a small range of notes whose ranges correspond to the notes within the usual voice ranges. 

As an architecture: 
- Deep Recurrent Neural Networks 
    - One summing up past information
    - Another summing up information coming from the future 
- A non-recurrent neural network for notes occurring at the same time.

After that, these three outputs are then merged and passed as the input of a fourth neural network whose output is probability.

Generation in dependency networks is performed using the pseudo-Gibbs sampling procedure. The advantage of this method is that we can enforce user defined constraints by tweaking Alg. 1:

![Alt Text](https://docs.google.com/uc?id=1ELpP-WYM1QGaAw5n6nwQNPQmPq2-5avK)

Their choice for the representation is so suitable with this algorithm. If they use piano roll representation, when they want to change of pitch of the value, one needs to change simultaneously a large number of variables (which is exponentially rare) because a long note is represented as the repetition of the same value over many variables. While this is achievable with only one variable change with our representation.

To understand how DeepBach perform, they build discrimination test. Subjects were presented series of only one musical extract together with the binary choice “Bach” or “Computer”. Fig. 5 shows how the votes are distributed depending on the level of musical expertise of the subjects for each model. 

For the comparision, they use Maximum Entropy Model (MaxEnt) and MultiLayer Perceptron (MLP)

_Ps. The Maximum Entropy model is a neural network with no hidden layer._


![Alt Text](https://docs.google.com/uc?id=1cvUGp_0TKZKPKLZvb6bNpTXjUoIhZjrv)


### 9) [Generating Polyphonic Music Using Tied Parallel Networks](http://www.hexahedria.com/files/2017generatingpolyphonic.pdf)

##### You can check [the blogpost](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/) which is the preliminary version of this paper.

_"We demonstrate training a probabilistic model of polyphonic music using a set of parallel, tied-weight recurrent networks, inspired by the structure of convolutional neural networks. This model is designed to be invariant to transpositions, but otherwise is intentionally given minimal information about the musical domain, and tasked with discovering patterns present in the source dataset."_

Polyphonic music has more complex nature than monophonic music. Both of them has sequential patterns between timesteps, however, polyphonic has simultaneous notes' harmonic intervals. Also musical structures exhibits transposition invariance. Transposition, in which all notes and chords are shifted into a different key, changes the absolute position of the notes but does not change any of these musical relationships. To capture the structure of chords and intervals in a transposition-invariant way, a neural network architecture would ideally consider relative positions of notes, as opposed to absolute positions.

Recurrent Neural networks are good at capturing of single-dimensional patterns. For the monophonic music, it works, however, for the polyphonic music, RNN can not capture the nature of the music. So that, as I summarized in second paper, some researchers use RBM with RNN to capture harmonical structure of simultaneous notes, however, it is not transposition-invariant. _(the RNN-RBM combines recurrent neural networks (RNNs), which can capture temporal interactions, and restricted Boltzmann machines (RBMs), which model conditional distributions.)_ Convolutional Neural Networks(CNN) are good at detect figures at different position in the picture. So that, if we can combine the invariance nature of CNN with RNN, we can model the polyphonic music better.

_"In the current work, we describe two variants of a recurrent network architecture inspired by convolution that attain transposition-invariance and produce joint probability distributions over a musical sequence. These variations are referred to as Tied Parallel LSTM-NADE (TP-LSTM-NADE) and Biaxial LSTM (BALSTM)."_

RBM's gradient is untractable. So that, researcher replace it with neural autoregressive distribution estimator(NADE), however, both RBM and NADE can not easily capture the relative relationship between inputs. Each transposed representation would have to be learned separately and this is not appropriate for musical structure. Convolutional neural networks address the invariance problem for images by convolving or cross-correlating the input with a set of learned kernels.

## ADD MORE INFO

### 10) [Combining LSTM and Feed Forward Neural Networks for Conditional Rhythm Composition](https://users.ionio.gr/~karydis/my_papers/MKPKarydisK2017%20-%20Combining%20LSTM%20and%20Feed%20Forward%20Neural%20Networks%20for%20Conditional%20Rhythm%20Composition.pdf)

As we know from the previous papers, LSTM is good at learning sequences, however, we can not add constraint to get conditional output via LSTM such as metrical structure or a given bass line. At that paper, researcher combine the LSTM and feed forward (conditinal) layers to compose long drum sequences based on external information, i.e. metric and bass information.

_"In this paper we introduce an innovative architecture for creating drum sequences by taking into account the drum generation of previous time steps along with the current metrical information and the bass voice leading."_

Let's look the architecture of the system.

- an LSTM network, corresponds to the drum representation.
    - Representation is based on binary. It represent snare, any tom event, open or closed hi-hats, and crash or ride cymbals.  For example, 10010 and 01010 represents a time step with simultaneous playing of kick and hi-hat followed by simultaneous playing of snare and hi-hat.

- the feedforward network, represents information of the bass movement, and the metrical structure information.
    - Moving on to the bass, we use information regarding the voice leading (VL) of bass. The first digit of this vector declares the existence of a bass or rest event, while the three remaining digits show the calculation of the bass voice leading in the following 3 different cases: [000] steady VL, [010] upward VL and [001] downward VL.
    - In addition to the bass information we included a 1x3 binary vector representing metrical information for each time step. This information ensures that the network is aware of the beat structure at any given point. 


![Alt Text](https://docs.google.com/uc?id=1gWlBMOEz0LgvqyM3fezJ3usAFm03ETco)

_"The proposed architecture consists of 2 separate modules for predicting the next drum event. An LSTM module learns sequences of consecutive drum events, while a feedforward layer takes information on the metrical structure and bass movement. The output of the network is the prediction of the next drum event."_

To understand the effect of feedforwards (conditional) layer, they build some experiment.

![Alt Text](https://docs.google.com/uc?id=1kUI_QnF-nqc0B5lFelhI4p9LFFSZIAaS)

![Alt Text](https://docs.google.com/uc?id=1102p78tM1Ks2EgCtx2aLpaEhVzOnsSWC)


_"Additionally, the preservation of a metrical structure in simple LSTM systems is only dependent on their ability to learn the metric structure these are trained on. The conditional layer enables the LSTM networks to simulate humans in both tasks: respond to changes in other instruments (e.g. bass) and ”tune-in” to certain metrical structures."_

**Main importance of this method is that you can add constraint to generated-composed music which is based on LSTM via feedforward(conditional) layer.**

### 11) [C-RNN-GAN: Continuous recurrent neural networks with adversarial training](https://arxiv.org/pdf/1611.09904.pdf)

##### This project is open-source. Please check their [source code.](https://github.com/olofmogren/c-rnn-gan)

##### To understand Generative Adversial Network (GAN), you can check [this blogpost.](https://skymind.ai/wiki/generative-adversarial-network-gan)

_"Our work represents tones using real valued continuous quadruplets of frequency, length, intensity, and timing. This allows us to use standard backpropagation to train the whole model end-to-end."_

At this paper, Olof Mogren propose the model which is a recurrent neural network with adversarial training. At the adversial training, we have 2 different RNN:

- Generator (G): Tries to generate the data that is indistinguishable from real data
    - The input to each cell in G is a random vector, concatenated with the output of previous cell.
- Discriminator (D): Tries to identify the data is generated or real
    - The discriminator consists of a bidirectional recurrent net, allowing it to take context in both directions
    into account for its decisions.

Both of them tries to achieve their goal. As we see, their goals' are completely opposite. So that, this is zero-sum game and when one improve, other one have to improve own skills. 

![Alt Text](https://docs.google.com/uc?id=1FSSM7IjpJa-UGTEeMiuvYJrsby9RDrxG)

To represent the musical data, they use 4 real valued scalars at each data point:
- Tone length
- Frequency
- Intensity
- Time

With this modeling, we can represent polyphonous chorus. Each tone is then represented with its own quadruplet of values as described above.

To improve the results:

- **Employ feature matching:** Encourage greater variance in the Generator. With that, we can avoid overfitting. "_Normally, the objective for the generator is to maximize the error that the discriminator makes, but with feature matching, the objective is instead to produce an internal representation at some level in the discriminator that matches that of real data."_

- **Freezing:** Which means stopping the updates of D whenever its training loss is less than 70% of the training loss of G. Because, at the training, they noticed that Discriminator can become too strong, resulting in a gradient that cannot be used to improve Generator. 

Also, they proposed modified version of C-RNN-GAN, at that version each LSTM cell can give output up to 3 tone. Name of this version is C-RNN-GAN-3. Benefit of this modification:

_"The generated music is polyphonous, but in the polyphony score in our experimental evaluation, measuring how often two tones are played at exactly the same time, C-RNN-GAN scored low. Allowing each LSTM cell to output up to three tones at the same time resulted in a model that scored much better with polyphony."_

Now, let's look the evaluation of baseline, C-RNN-GAN and C-RNN-GAN-3.

##### "Baseline: Our baseline is a recurrent network similar to our generator, but trained entirely to predict the next tone event at each point in the recurrence."


![Alt Text](https://docs.google.com/uc?id=1WxxzoatGb0byp0SBeg_eAjHtVyfMkJc9)

##### For more info about the metrics and explanations, please check the [original paper.](https://arxiv.org/pdf/1611.09904.pdf)

### 12) [MIDINET: A CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORK FOR SYMBOLIC-DOMAIN MUSIC GENERATION](https://arxiv.org/pdf/1703.10847.pdf)

##### This paper is based on open source code. Please check [their repo.](https://github.com/RichardYang40148/MidiNet)

##### [Listen samples.](https://richardyang40148.github.io/TheBlog/midinet_arxiv_demo.html)

Their method is combination of CNN and GAN. 

- They use CNN+GAN to generate melody via one bar after another bar in symbolic domain as MIDI.
- With GAN, they can control-manipulate the output.

In this method, generation is successive. They generate one bar after another one. So that, they can use 2D matrix to represent notes over time. With that, they can generate melody in MIDI format.

Their GAN architecture consists of 3 different CNN: 
- Generator CNN
    - Takes random noise as an input and transform into 2-D score like representation.
- Discriminator CNN
    - Predicts whether it is from a real or a generated MIDI.
- Conditioner CNN
    - We can not impose our conditions with just Generator and Discriminator. With Conditioner, we can use previous bars as a condition of generation of the present bar. With this CNN, model can "look back" without RNN and can generate arbitrary number of bars. Also, thanks to Conditioner CNN, we can generate the music based on the determined chord progression or based on the few starting notes.

![Alt Text](https://docs.google.com/uc?id=161Kz4MqWh3VEKOttvBhnOAD9ZwC5-I9F)

Comparision of the MIDINET with other generation methods

![Alt Text](https://docs.google.com/uc?id=1d_PfbdeEKp67CmmrHYA24uWs9NBvgaSr)

Let's dive into the details of methods. :)

- **For representation**, they use one-hot piano roll based representation. They omit the velocity and system can not distinguish one long note and two short note repeating notes. 

- **Generator and Discriminator CNN** is based on DCGAN (Deep Convolutional Generative Adversial Network). 
    - Generator tries to generate the data that is indistinguishable from real data. Thanks to _transposed convolutional layers_, generator can upsamples smaller vectors/matrices into larger one.
    - Discriminator tries to identify the data is generated or real

    For the training, they also employ feature matching like C-RNN-GAN method.

- To condition the generation, available prior knowledge can impose via vector which encode the information. After the reshape to appropriate format, we can add this vector into different layers of Generator or Discriminator. This is 1D Conditions. (You can check Figure 1) With this type of condition, model can generate current note based on previous notes. We can directly add such a conditional matrix to the input layer of D to influence all the subsequent layers. For the 2D Conditions, we need to shape 2D conditional matrix into different smaller appropriate formats as you can see from figure 1.  Conditioner and Generator use exactly same shape filters. Thus, we can add conditional info into Generator via vectors. So that, we can influence the generation.

To control the creativity

- Capitalize the effect of feature matching. With that, generated music will be more similar to real existing music.

- By restricting the conditioning by inserting the conditioning data only in the intermediate convolution layers of G.

They use 1022 MIDI tabs of pop music from [TheoryTab.](https://www.hooktheory.com/theorytab) These files include both melody and underlying chord progression.

- For chords, they use 13 dimensional representation (last dimension represent the key) instead of 24 dimensional one hot vector.

![Alt Text](https://docs.google.com/uc?id=1B2UEsR3C90r-TMFB_f7O7dpNxCS1Ep0b)

-  _"Moreover, for simplicity, we shifted all the melodies into two octaves, from C4 to B5, and neglected the velocity of the note events."_

### 13) [A Unit Selection Methodology for Music Generation Using Deep Neural Networks](https://arxiv.org/pdf/1612.03789.pdf)

At that paper, researcher use _concatenation_ to generate music which process is based on _unit selection_. This idea comes from Text-To-Speech (TTS) systems. At that system, small audio (speech) units are concenated to get long natural-like speech. When apply this idea directly to music, we have some challenges.

- Output is restricted to what is available in the unit library. 
- Concatenation process can generate _jumps_ or _shifts_ and it can cause to unpleasent output.

The method is based on 2 step.
- Unit selection based reconstruction for Deep-AutoEncoder
- Concenate the units based on ranks.
    - Semantic Relevance Score which is based on difference of two units in embedding space which is created by deep structured semantic model.(DSSM)
    - Concatenation Costs which is determined by LSTM which evaluate the likelihood of two consecutive units.

**Reconstruction Using Unit Selection**

They create 80 million unique measures to augment their dataset which contains 170.000 unique measures. For the augmentation, they manipulate pitches via linear shifts(transpositions) and alterations of the intervals between notes. After the augmentation, they represent the data via Bag-of-Words (BOW) like feature vector. This results in 9,675 actual features (parameters). The feature vector include counts of note tuples, counts of pitch class etc. Each unit is described (indexed) as a 9,675 size feature vector.

![Alt Text](https://docs.google.com/uc?id=1HBuRYPZ26shjRfoOaDDL8sjMQ8Ib7yOj)

Reconstruction has 2 steps:
- **feature vector reconstruction**: performed by _decoder_
- **music reconstruction**: the process of selecting a unit that best represents the initial input musical unit
    - Output of the decoder may not be in the library. So that, we need to choose the unit which is the most similar from the library.

Thanks to autoencoder, we have embeddings and we will use these for the generation.

**Generation using Unit Selection**

![Alt Text](https://docs.google.com/uc?id=1tYMxwIyq7h0PncFcxIu03on0yRyWGPnF)

- For semantic relevance -> DDSM based BOW-like features
- Concatenation cost -> LSTM which evaluate the likelihood of two consecutive unit

With DDSM, we can measure the similarity between units, however, it can not provide which one should come first. 

_"In an attempt to ensure that the music remains valid after combining new units we employ a concatenation cost to describe the quality of the join between two units. This cost requires sequential information at a more fine grained level than the BOW-DSSM can provide."_ To learn sequential information (context), LSTM is used. 

**Ranking process** includes 4 steps:
- Rank all units according to similarity (semantic relevance) at the embedding space according to input seed
- Take top %5 of them and re-rank according to concatenation cost with the input
- Re-rank the same top 5% based on their combined semantic relevance and concatenation ranks.
- Select the unit with the highest combined rank.

**Evaluate the Model**

We can evaluate the model using a ranking test. The task is that predict the next unit given input which is not in the training set. For the prediction, 50 candidates (1 ground truth, 49 randomly selected unit) are ranked. After that, accuracy is calculated. 

![Alt Text](https://docs.google.com/uc?id=1Zytg0aEPBrYM7lxWT03zwVPl1Ao8YN69)

The main issue with this evaluation, it can not evaluate likealibity of the music. Because, one of the randomly selected unit which is selected as the best from the ranking system can be more appropriate as a next unit for a give unit. So that, subjective evaluation has been done.

![Alt Text](https://docs.google.com/uc?id=1KIPBCKiOM1SlzHaMcAzdAJl8hiMaXsOZ)

### 14) [TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING](https://affect.media.mit.edu/pdfs/17.Jaques-Tuning.pdf)

##### To listen samples, you can check [this drive folder.](https://drive.google.com/drive/folders/0BycMAUU0mKhwN3BEMENCMXN2cFE)

The objective of this method is that control the generation of melodies with user constraints. 

For the reward part of the Reinforcement Learning (RL), they train the LSTM which called as Note-RNN. 

_"Our research question is therefore whether such music-theory-based constraints can be learned by an RNN, while still allowing it to maintain note probabilities learned from data."_

For this question, they propose _RL Tuner_. They tries to impose structure to RNN via Reinforcement Learning. The reward function is combination of task-related proporties (which can be user constraint) and likelihood function from pre-traines LSTM which is trained on a large corpus of songs to predict the next note in a musical sequence. (which learns the context) Thus, when we preserve the context thanks to RNN, we can impose some constraint via RL. So that, there is trade-off between influence of data (context) and heuristic reward (user constraint).

###### Ps. This paper has made different contributions for the ML-DL area, however, most of them is not direclty related with music generation. So that, I will skip these parts. If you are curious especially for the RL, please read [the paper](https://affect.media.mit.edu/pdfs/17.Jaques-Tuning.pdf).
 

The goal is that combine:
- Concepts from music theory 
- Melodic structure which is learned from data via LSTM.

To accomplish this task, they propose _RL Tuner_ which consists of 2 RNN and 2 deep Q network.
- **Note RNN**: To supply the initial weights which is learned from data via LSTM for another parts of the system
- **Reward RNN**: Fixed copy of Note RNN. Is used to supply part of the reward value used to train the model, and is held fixed during training.
- **Q Network**:  Learn to select next note (next action a) from the generated (partial) melody so far (current state a). 
- **Target Q Network**: Which estimates the value of the gain and which has been initialized from what the Note RNN has learnt.

"_The Q Network’s reward r combines two measures:_

– _adherence to what has been learnt, by measuring the similarity with the note predicted by the Reward RNN recurrent network;_

– _adherence to user-defined constraints (in practice according to some musical theory rules, e.g., consistency with current tonality, avoidance of excessive repetitions etc.), by measuring how well they are fulfilled._"

![Alt Text](https://docs.google.com/uc?id=16vQPq1sIqLGn3l5_497SbOl37J9ohsUr)

**Experiments**

![Alt Text](https://docs.google.com/uc?id=1EQDE8U2MGJ-vCLikMie5RBDISsUmqVNL)
![Alt Text](https://docs.google.com/uc?id=1KwEXzE-JoAdDXUEiuftyzxdnab3353t3)
![Alt Text](https://docs.google.com/uc?id=1B7Mh0KIoyt-JYHiep9bHVjN5bvafuEzB)
