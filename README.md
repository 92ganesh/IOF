# IOF
IOF stands for Image Orientation Fixer. It is an android app to auto fix mis oriented images. Many a times you might have faced problem that pics taken from mobile are not prefectly oriented. They might be little tilted or completely inverted.
Your phone automatically decides orientation based on how you hold it using accelerometer. This method works perfectly if your taking pics of objects in front of you but does not give fails while taking pics of object lying flat on ground e.g. while you are taking pics of any document lying on table. It is because your camera is configured to respond to accelerometer's X and Y axes. 
Hence your phone can't determine appropriate image orientation.

This is one of the problem faced by students in day of day life as we need to take pics of notes, books, news articles etc

## Tech Stack
- Android studio 3
- python 2.7
- tf.Keras (python)
- tensorflow lite
- openCV (java)

## How to use
- If you have android phone having android ver as 4.4 and above and directly install the latest apk (having highest version number) present in folder called *releases*
- for android 4.3 and below, app is not tested but may work
- Allow storage access permission at app startup
- App will create folder called *IOF* in your sdcard for storing data
<span style="color:red">warning:</span> please make sure you dont have any folder named IOF there already otherwise it will be overwritten
- place images needs to be fixed in folder called *Pages* inside *IOF* folder. 
- (Optional mode) if you want to fix images having only one character in image (character dataset) do the same for individual character to be predicted but remember to keep them in folder called *Character* instead of *Pages*
- Press run
- Your result will be availabe in folder called *fixed* inside *IOF* folder
<span style="color:blue">Note:</span> The app will be performing lot of complex computation hence wont be available for usage for a while. Please have patience and dont close the app if you see a white screen after pressing run. Result will be available once every image has been fixed.

