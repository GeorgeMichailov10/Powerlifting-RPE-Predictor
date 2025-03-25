1. Test and figure out which libraries will be used for video metric extraction. (OpenCV, MediaPipe, OpenPose)
2. Develop basic prototype video processer that works reliably.
3. Create pipeline to extract from instagram videos or youtube pl livestreams.
   -> People often put the rpe in their instagram videos and can search their names on openpowerlifting for some reference.
   -> Extract as much information as possible including reps, weight, rpe, speed, comp maxes from openpl, bodyweight, etc. Ok to have blanks.
   -> Flag people who have competed recently since these will have the most accurate markers.
4. Clean and normalize from the extracted data. Might run over it with an LLM to find if any data doesn't make sense.
5. Train a generalist model with these metrics to take input and spit out a max for an rpe and reps or predicted max for the day.
6. Verify model accuracy using professional powerlifters and people who have competed recently
7. Create and train the small specialist model on my own data to see what is needed to get it correctly predicting for myself.
8. Build mobile app (should be extremely simple UI and process)
