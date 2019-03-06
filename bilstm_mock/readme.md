### This is a simple example to reproduce our results for Bi-LSTM Model. 

- Unzip our txt files.
  
    ```bash
    unzip txt.zip
    ```
- You should run preprocess.py to generate our input and output sequences.

    ```bash
    python preprocess.py <True or False> <Sequence Length>
    ```
    Example:

    ```bash
    python preprocess.py True 10
    ```

- Train our model.
    ```bash
    python train_model.py <rnn size> <learning rate> <batch size> <number of epochs>
    ```
    Example:

    ```bash
    python train_model.py 128 0.001 32 10
    ```

- Now, we can sample notes. 
    ```bash
    python sample_sequences.py <sequence length > <rnn size> 
    ```
    Example:

    **These parameters have to be equal with previous sequence length and rnn size**
    ```bash
    python sample_sequences.py 10 64
    ```