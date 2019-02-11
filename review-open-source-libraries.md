## Review of Open Source Libraries

1)[Melody RNN](https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn)

This project is based on Google's Magenta. They introduce 3 different models in Melody RNN. [(For more info)](https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/)
- Basic RNN
  - _"The input to the model was a one-hot vector of the previous event, and the label was the target next event. The possible events were note-off (turn off any currently playing note), no event (if a note is playing, continue sustaining it, otherwise continue silence), and a note-on event for each pitch (which also turns off any other note that might be playing)."_ 
- Lookback RNN
  - _"Lookback RNN introduces custom inputs and labels. The custom inputs allow the model to more easily recognize patterns that occur across 1 and 2 bars. They also help the model recognize patterns related to where in the measure an event occurs. The custom labels make it easier for the model to repeat sequences of notes without having to store them in the RNN’s cell state."_
- Attention RNN
    - _"We just always look at the outputs from the last 𝑛 steps when generating the output for the current step. The way we “look at” these steps is with an attention mechanism"_
  
_Lookback RNN_ and _Attention RNN_ tries to solve problem which is about capture the long term structure of music.

They have docker container for models. We can use pretrained models or directly create our own model. 

_For pretrained models, we can supply this models with primer sequence which can be python list or midi._

Now, I would like to focus on how we [create our own model.](https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn#train-your-own)

**How to train our model?**
- **Install Magenta environment** [(Source)](https://github.com/tensorflow/magenta/blob/master/README.md)
  
    We can use Conda or Docker.

    **_Conda:_**
    ```sh
    curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
    bash /tmp/magenta-install.sh
    ```
    After installation, active it each time.
    ```sh
    source activate magenta
    ```
    **_Docker:_**
    ```sh
    docker run -it -p 6006:6006 -v /tmp/magenta:/magenta-data tensorflow/magenta
    ```


- **Create our own note sequence from MusicXML or midi.** [(Source)](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md)

    ```sh
    INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>

    # TFRecord file that will contain NoteSequence protocol buffers.
    SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

    convert_dir_to_note_sequences \
    --input_dir=$INPUT_DIRECTORY \
    --output_file=$SEQUENCES_TFRECORD \
    --recursive
    ```

- **Create SequenceExamples**
  
    We need to prepare our dataset as input and output. SequenceExamples represent both our input and output. 
    ```sh
    melody_rnn_create_dataset \
    --config=<one of 'basic_rnn', 'lookback_rnn', or 'attention_rnn'> \
    --input=/tmp/notesequences.tfrecord \
    --output_dir=/tmp/melody_rnn/sequence_examples \
    --eval_ratio=0.10
    ```
- **Train and Evaluate Model**
    
    Before generation, we need to train and evaluate our model. There are different hyperparameters, train and evaluation options. We will discuss it later. For instance:
    ```sh
    melody_rnn_train \
    --config=attention_rnn \
    --run_dir=/tmp/melody_rnn/logdir/run1 \
    --sequence_example_file=/tmp/melody_rnn/sequence_examples/training_melodies.tfrecord \
    --hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
    --num_training_steps=20000
    ```

- **Generate Melodies**
    
    We can generate melodies during training or after training. Again, there are different hyperparameters which can be optimized.
    ```sh
    melody_rnn_generate \
    --config=attention_rnn \
    --run_dir=/tmp/melody_rnn/logdir/run1 \
    --output_dir=/tmp/melody_rnn/generated \
    --num_outputs=10 \
    --num_steps=128 \
    --hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
    --primer_melody="[60]"
    ```

    **As I wrote before, we can initialize generation via Python List or midi file. It can be so helpful for us to understand how generated melodies changes with seeding. From there, we can make comment about how makam-usul of the generated song be related with seeding part.**

2)[Music VAE](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae)

This project is also comes from Magenta Team. For technical details, please refer the [blogpost](https://magenta.tensorflow.org/music-vae) and [paper](https://goo.gl/magenta/musicvae-paper).

As as summary, for short sequences (e.g., 2-bar "loops"), they use a bidirectional LSTM encoder and LSTM decoder. (Bi-LSTMs allow the process sequences in forward and backward directions, making use of both past and future contexts.) For longer sequences, they use a novel hierarchical LSTM decoder, which helps the model learn longer-term structures.

Representation of melodies is same with Melody RNN and [Drums RNN.](https://github.com/tensorflow/magenta/tree/master/magenta/models/drums_rnn) (Note that, Drums RNN creates polyphonic drum tracks based on combination of monophonic generation. As our Turkish Music Generation case, we are working on monophonic generation. So that, I won't cover Drums RNN here.)

There are different ways to reproduce their works.
- [Colab Notebook](https://g.co/magenta/musicvae-colab)
- Generate Script with Pre-Trained Models
  - Download Pre-Trained weights from [here](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae#pre-trained-checkpoints)
  - Now we have 2 options
    - Sample
      - Sampling decodes random points in the latent space of the chosen model and outputs the resulting sequences in output_dir
        ```sh
        music_vae_generate \
        --config=hierdec-trio_16bar \
        --checkpoint_file=/path/to/music_vae/checkpoints/hierdec-trio_16bar.tar \
        --mode=sample \
        --num_outputs=5 \
        --output_dir=/tmp/music_vae/generated
         ```
  
    - Interpolate
      - To interpolate, you need to have two MIDI files to inerpolate between. There are certain constraints about interpolation. We can use *mel_2bar models* for our generation. Because, it works with only if the input files are exactly 2-bars long and contain monophonic non-drum sequences.
        ```sh
        music_vae_generate \
        --config=cat-mel_2bar_big \
        --checkpoint_file=/path/to/music_vae/checkpoints/cat-mel_2bar.ckpt \
        --mode=interpolate \
        --num_outputs=5 \
        --input_midi_1=/path/to/input/1.mid
        --input_midi_2=/path/to/input/2.mid
        --output_dir=/tmp/music_vae/generated
        ```
- Train our own model

  - **Install Magenta environment** [(Source)](https://github.com/tensorflow/magenta/blob/master/README.md)
    
      We can use Conda or Docker.

      **_Conda:_**
      ```sh
      curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
      bash /tmp/magenta-install.sh
      ```
      After installation, active it each time.
      ```sh
      source activate magenta
      ```
      **_Docker:_**
      ```sh
      docker run -it -p 6006:6006 -v /tmp/magenta:/magenta-data tensorflow/magenta
      ```


  - **Create our own note sequence from MusicXML or midi.** [(Source)](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md)

      ```sh
      INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>

      # TFRecord file that will contain NoteSequence protocol buffers.
      SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

      convert_dir_to_note_sequences \
      --input_dir=$INPUT_DIRECTORY \
      --output_file=$SEQUENCES_TFRECORD \
      --recursive
      ```
  - Train it
    ```sh
    - music_vae_train \
    --config=cat-mel_2bar_small \
    --run_dir=/tmp/music_vae/ \
    --mode=train \
    --examples_path=/tmp/music_vae/mel_train_examples.tfrecord
    ```
   