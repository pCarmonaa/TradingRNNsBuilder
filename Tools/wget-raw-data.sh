#!/bin/sh
initDate=$(date +"%y%m%d" -d 2015-09-01)
lastDate=$(date +"%y%m%d" -d 2020-02-10)
folderToDownload="./RawData"

mkdir -p $folderToDownload
while [ $lastDate -ge $initDate ]
do
	wget http://www.megabolsa.com/cierres/$initDate.txt -P $folderToDownload
	initDate=$(date +"%y%m%d" -d "$initDate + 1 days")
done
