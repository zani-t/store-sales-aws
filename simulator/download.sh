#!/bin/bash

kaggle competitions download -c store-sales-time-series-forecasting
unzip store-sales-time-series-forecasting.zip -d simulator/store-sales-time-series-forecasting
rm store-sales-time-series-forecasting.zip