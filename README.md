This is the architecture <br>

<p align="left">
  <img src="https://raw.githubusercontent.com/luifire/change_detection_in_sar_images/main/submissions/impressions/architecture.png" height=300>
</p>

(Cleaned) image of day 0 <br>
<p align="left">
  <img src="https://raw.githubusercontent.com/luifire/change_detection_in_sar_images/main/submissions/impressions/amrym_pred_interest.png" height=200>
</p>

predicted image of day 6 <br>
<p align="left">
  <img src="https://raw.githubusercontent.com/luifire/change_detection_in_sar_images/main/submissions/impressions/amrym_pred_future.png" height=200>
</p>

Difference of the whole image. <br>
<p align="left">
  <img src="https://raw.githubusercontent.com/luifire/change_detection_in_sar_images/main/submissions/impressions/diff_image.png" height=200>
</p>


---------------------
Due to space constraints I couldn't upload all images and other files from this repo.

Note that 2 teams worked on the same problem, so in some submissions one will find the other teams result.
Our team is Team-A / Team-1.

---------------------

Code description:
Global_Info:
The main file is called Global_Info
It contains all the settings that you can change.
They are usually described significantly enough for you to understand.

create_dataset: -contains main
will create the dataset for you

CreateDatasetWorker:
Don't ask why, I threaded the creation of the database.

cut_water: -contains main
will cut of water of some given images

evaluate_period: -contains main
will evaluate an entire period and create a CSV file with mse and so on for that period

evaluation: -contains main
will evaluate file pairs for you.
It will create the diff image and predicted image and so on in the folder:
branches/#branch_name/predictions

file_pair_creator:
Has multiple ways of creating file pairs for you.
Note that under database/cleaned_data.csv are the cleaned data for us 
(the once that are not shifted).

Helper:
Some small helper functions

image_manipulation:
Some small helper functions to manipulate images

layer_block_creator:
Some small helper functions to create the ley_net

ley_net: -contains main
Creates the ANN.

load_database:
Loads the database for training.

NormalizeDate:
Normalizes the data.

PointsAndRectangles:
Some computation for points and rectangles.
copied and modified from https://wiki.python.org/moin/PointsAndRectangles

resnet:
copied from
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py

resume_training: -contains main
Resumes the training

---------------------
Git Description:

Branches start with increasing numbers, then follows a small description.
In Repo you will find a document called Experiments.txt which has a small summation of everything I did.

---------------------
Folder description:

Branches:
Here you find results of the given experiment. 
	In the subfolder models you can find the used model, an image of the model and maybe some in between results.
	In the subfolder predictions you can find important predictions for this experiment.

Database:
Data can be obtained directly from me or you run the creation of data again.
Use code instructions to do so.
	
lui_net:
you can find the newest code. (I wanted to rename it to ley_net, but PyCharm didn’t want me to).
