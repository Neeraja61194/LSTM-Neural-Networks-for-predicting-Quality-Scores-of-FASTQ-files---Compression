LSTM Neural Networks for predicting Quality Scores of FASTQ files 

Outcome: Compression of FASTQ files using LSTM for predicting quality scores based on Machine Information (Lane, Tile, X and Y coordinates).

There are 3 basic codes (Preprocessing, Training and Testing) to generate the final compressed predicted FASTQ File. 

• Preprocessing step takes a FASTQ file as input and outputs a CSV File.

Run the following command to do preprocessing:

    python3 Preprocessing.py test.fq

The “test.fq” can be replaced by the input file of your choice.
Output: train_data.csv

• Training Step takes the output of Preprocessing step, the “train_data.csv” as the input and trains the LSTM model. The best model checkpoints of weights are stored in a “model_checkpoints” folder.

Run the following command to do training:

    python3 Training.py train_data.csv

The “train_data.csv” is the output file obtained after running the preprocessing step. Outputs , which are the best model weights after each epoch, are stored in “model_checkpoints” folder.
Output: Model Weights

• Testing Step takes 3 input arguments. The first argument is the input file, second is the “train_data.csv” from Preprocessing and the third is the best model weight that you want to use for prediction. This will output a final “predicted_qual_score.txt” and a compressed “predicted_qual_score.txt.gz”. These can be tested against the corresponding baseline quality score for size variations.

Run the following command to do testing:

    sh exec_testing.sh test.fq train_data.csv model_checkpoints/weightsimprovement-50-0.0334.hdf5

Output: predicted_qual_score.txt, predicted_qual_score.txt.gz