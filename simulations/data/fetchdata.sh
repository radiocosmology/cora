#!/bin/bash

datadir=$(dirname $0)

data_url="http://www.cita.utoronto.ca/~jrs65/cyl_skydata.tar.bz2"

echo "Downloading data file....."

wget $data_url

RETVAL=$?
[ $RETVAL -eq 0 ] && echo "Done."
[ $RETVAL -ne 0 ] && echo "Download failed, you're on your own.\nTry fetching $data_url manually."


echo "Unpacking data files..."

tar xvjf cyl_skydata.tar.bz2

RETVAL=$?
[ $RETVAL -eq 0 ] && echo "Done."
[ $RETVAL -ne 0 ] && echo "Unpacking failed, you're on your own."



