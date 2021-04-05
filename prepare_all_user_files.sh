#!/bin/bash
START_TS="$(date +"%s")"

COUNT=0
for TYPE in "train" "test" "val"
do
	for FILE in processed/$TYPE/*.feather
	do

	     echo $FILE
	     OUTPUT=`dirname $FILE`/ready/`basename $FILE`
	     python prepare_comments.py $FILE $OUTPUT
	     COUNT=$(( COUNT + 1 ))
	     if [ $(( COUNT % 10 )) -eq 0 ]
	     then
		     THIS_TS="$(date +"%s")"
		     TOOK=$(( THIS_TS - START_TS ))
		     echo "$COUNT FILES PROCESSED in $TOOK seconds"
	     fi
	done
done	

echo "DONE! $COUNT FILES PROCESSED in $TOOK seconds"
