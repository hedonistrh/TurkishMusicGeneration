### TurkishMusicGeneration

To get same environment and reproduce results easily, we are using [Conda](https://conda.io/en/latest/).

- Download latest version of installer from [repo](https://repo.continuum.io/archive/).
  
    *For Linux*
    ``` sh
    wget -c https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
    chmod +x Anaconda3-2018.12-Linux-x86_64.sh
    bash ./Anaconda3-2018.12-Linux-x86_64.sh -b -f -p /usr/local
    ```

    *For MacOs*
    ###### This is based on [this blogpost](https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072).
    ``` sh
    # Go to home directory
    cd ~
    # You can change what anaconda version you want at 
    # https://repo.continuum.io/archive/
    curl https://repo.continuum.io/archive/Anaconda3-2018.12-MacOSX-x86_64.sh -o anaconda3.sh
    bash anaconda3.sh -b -p ~/anaconda3
    rm anaconda3.sh
    echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bash_profile 
    # Refresh basically
    source .bash_profile
    conda update conda
   ```

    After the installation, to test, you can open your terminal and try to open jupyter notebook.

    ``` sh
    jupyter notebook
    ```

- To use same conda environment:
    ``` sh
    conda env create -f environment.yml
    ```

- To activate your environment:
    ``` sh
        source activate turkishMusicGen
    ```

- To deactivate your environment:
    ``` sh
        source dectivate 
    ```

- How to update our environment with new(or updated) .yml file ([source1](https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file), [source2](https://stackoverflow.com/questions/45510430/install-packages-into-existing-conda-environment-specified-in-environment-yml)):
    ``` sh
    source activate turkishMusicGen
    conda env update -f=environment.yml
    ``` 

    or

    ``` sh
    conda env update --file environment.yml
    ``` 

    
[For more info](https://towardsdatascience.com/environment-management-with-conda-python-2-3-b9961a8a5097).