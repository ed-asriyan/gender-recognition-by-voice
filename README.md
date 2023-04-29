# Gender Recognition using Voice
The script recognize gender by voice in .wav file

## Command-line
### Preparation
Build docker image:
```commandline
docker build --target cli -t gender-recognition .
```

### Run
```commandline
docker run --rm -v <path_to_dir_wav_file>:/source /source/<wav_file_name>.wav
```
