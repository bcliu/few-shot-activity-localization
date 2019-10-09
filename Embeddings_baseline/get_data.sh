#!/bin/bash

mkdir jpegs_256
cd jpegs_256/

wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

cat ./ucf101_jpegs_256.zip* > ./ucf101_jpegs_256.zip

rm ./jpegs_256/ucf101_jpegs_256.zip.001 ./jpegs_256/ucf101_jpegs_256.zip.002 ./jpegs_256/ucf101_jpegs_256.zip.003

no | unzip jpegs_256/ucf101_jpegs_256.zip

cd ../

