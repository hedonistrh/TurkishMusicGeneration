## Experiments

### Magenta RNN

- I have started with Conda, however, it requires miniconda2 and I was not able to get correct environment. So that, I started to use Docker.
  
- Firstly, I installed Docker from [official website.](https://www.docker.com)
- Now we can get Docker environment for Magenta via this script.
    ```sh
    docker run -it -p 6006:6006 -v /tmp/magenta:/magenta-data tensorflow/magenta
    ```
- You can check docker containers via
    ```sh
    docker container ls -a
    ```
- You can stop it via 
    ```sh
    docker container stop <container_id>
    ```
- Tou can delete it via
    ```sh
    docker rm -v <container_id>
    ```
    - **Caution** You can prune all docker and related volumes with
        ```sh
        docker image prune -a
        ```
- Test the installation via this script. This will create 10 .midi file in _/tmp/magenta/lookback_rnn/generated_ You can change this directory via change the argument of *output_dir* and number of generated .midi file via argument of *num_outputs*. Also, there are another parameters, you can change those.
  ```sh
  melody_rnn_generate \
  --config=lookback_rnn \
  --bundle_file=/magenta-models/lookback_rnn.mag \
  --output_dir=/magenta-data/lookback_rnn/generated \
  --num_outputs=10 \
  --num_steps=128 \
  --primer_melody="[60]"
  ```
- [According to the documentation](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md), we can use MusicXML files or .midi files as input. In SymbTR, we have both of them. Lets download it from SymbTR. You can directly clone it or use [DownGit](https://minhaskamal.github.io/DownGit/#/home)
    ```sh
    git clone https://github.com/MTG/SymbTr.git
    ```
- Now, we need to convert our datas into appropriate format via 
  ```sh
    INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>

    # TFRecord file that will contain NoteSequence protocol buffers.
    SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

    convert_dir_to_note_sequences \
    --input_dir=$INPUT_DIRECTORY \
    --output_file=$SEQUENCES_TFRECORD \
    --recursive
    ```
    - Start with MusicXML, I have copied folder into /tmp/magenta _(You can use any file manager to copy it)_
    
     ```sh
    INPUT_DIRECTORY=/magenta-data/MusicXML

    # TFRecord file that will contain NoteSequence protocol buffers.
    SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

    convert_dir_to_note_sequences \
    --input_dir=$INPUT_DIRECTORY \
    --output_file=$SEQUENCES_TFRECORD \
    --recursive
    ```
    This gives an warning for each file. For instance:

    > WARNING:tensorflow:Could not parse MusicXML file /magenta-data/MusicXML/hicaz--sarki--aksak--kis_geldi--sevki_bey.xml. It will be skipped. Error was: Could not find fifths attribute in key signature.

    - So that, I have tried .midi files from SymbTR. _(I took 1000 of them)_
    ```sh
    INPUT_DIRECTORY=/magenta-data/midi

    # TFRecord file that will contain NoteSequence protocol buffers.
    SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

    convert_dir_to_note_sequences \
    --input_dir=$INPUT_DIRECTORY \
    --output_file=$SEQUENCES_TFRECORD \
    --recursive
    ```

    It works. :) For instance
    > INFO:tensorflow:Converted MIDI file /magenta-data/midi/beyati--sarki--devrihindi--nar-i_firkat--mahmud_celaleddin_pasa.mid.

- Now, lets generate our dataset via one of the model. 
    ```sh
    melody_rnn_create_dataset \
    --config=<one of 'basic_rnn', 'lookback_rnn', or 'attention_rnn'> \
    --input=/tmp/notesequences.tfrecord \
    --output_dir=/tmp/melody_rnn/sequence_examples \
    --eval_ratio=0.10
    ```

    - I would like to go with _Attention Rnn._

    ```sh
    melody_rnn_create_dataset \
    --config=attention_rnn \
    --input=/tmp/notesequences.tfrecord \
    --output_dir=/tmp/melody_rnn/sequence_examples \
    --eval_ratio=0.10
    ```

    It discards some of our input. You can find .tfrecord of our input and output at _/tmp/melody_rnn/sequence_examples_ It correctly produces 395 output from 1000 input file.
    
    **We should check this error.**
    > INFO:tensorflow:DAGPipeline_MelodyExtractor_training_polyphonic_tracks_discarded: 217

    ![Alt Text](https://docs.google.com/uc?id=1JCLpC3a4cMhaMYZiL-9kXntEsUNe4WeY)

    After that, we have 
    > - 3.8Mb eval_melodies.tfrecord   
    > - 45Mb training_melodies.tfrecord

- [Time to train our model](https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn#train-and-evaluate-the-model)
    ```sh
    melody_rnn_train \
    --config=attention_rnn \
    --run_dir=/tmp/melody_rnn/logdir/run1 \
    --sequence_example_file=/tmp/melody_rnn/sequence_examples/training_melodies.tfrecord \
    --hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
    --num_training_steps=20000
    ```

    - We can watch training via _TensorBoard_
        ```sh
        docker exec <container_id> tensorboard --logdir=/tmp/melody_rnn/logdir        
        ```
        Then check http://localhost:6006


- Finally, we can [generate our melodies](https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn#generate-melodies)
  ```sh
    melody_rnn_generate \
    --config=attention_rnn \
    --run_dir=/tmp/melody_rnn/logdir/run1 \
    --output_dir=/magenta-data/generated \
    --num_outputs=10 \
    --num_steps=128 \
    --hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
    --primer_melody="[60]"
    ```

    You can see your outputs at _/tmp/magenta/generated_