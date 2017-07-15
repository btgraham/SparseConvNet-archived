Download and unzip train.zip and test.zip

#Deal with a few unprocessable images
rm train/492_*
cp test/25313_left.jpeg test/25313_right.jpeg
cp test/27096_left.jpeg test/27096_right.jpeg

Download
https://www.kaggle.com/c/diabetic-retinopathy-detection/download/trainLabels.csv.zip
and 
https://www.kaggle.com/blobs/download/forum-message-attachment-files/2877/retinopathy_solution.csv
and 
create links to the training and test data into train01234/*/* and testPrivate/*/* and testPublic/*/* using commands:

for a in $(grep -v image trainLabels.csv|grep -v ^492_); do b=`echo $a|cut -d , -f 1`; c=`echo $a|cut -d , -f 2`; ln -s ../../train/$b.jpeg train01234/$c/$b.jpeg;  done
for a in $(grep Private retinopathy_solution.csv); do b=`echo $a|cut -d , -f 1`; c=`echo $a|cut -d , -f 2`; ln -s ../../test/$b.jpeg testPrivate01234/$c/$b.jpeg;  done
for a in $(grep Public retinopathy_solution.csv); do b=`echo $a|cut -d , -f 1`; c=`echo $a|cut -d , -f 2`; ln -s ../../test/$b.jpeg testPublic01234/$c/$b.jpeg;  done

Then run preprocessImages.py
