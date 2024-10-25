#!/usr/bin/env bash
set -e

S3_URL=s3://jojjiw/satellite_ma/llamaindex/02_tha_cloud/.env

function get {
    echo Retrieve .env from S3
    aws s3 cp $S3_URL .env
}

function save {
    echo Saving .env to S3
    aws s3 cp .env $S3_URL
}

if [ -z "$1" ]; then
    echo "Usage: $0 {get|save}"
    exit 1
fi

case $1 in
    get)
        get
        ;;
    save)
        save
        ;;
    *)
        echo "Invalid argument. Usage: $0 {get|save}"
        exit 1
        ;;
esac
