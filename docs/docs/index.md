# Water Potability Prediction documentation!

## Description

Welcome! This project demonstrates an end-to-end MLOps workflow to predict water potability using machine learning. We use tools like MLflow for tracking, DVC for versioning, and Tkinter for creating a desktop application.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://bucket-name/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://bucket-name/data/` to `data/`.


