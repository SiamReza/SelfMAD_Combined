import os
import logging
import csv
import re
from datetime import datetime

# a function  to create and save logs in the log files
def log(path, file, csv_logging=True):
    """[Create a log file to record the experiment's logs]

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
        csv_logging {bool} -- whether to create a CSV version of the log file

    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    # console_logging_format = "%(levelname)s %(message)s"
    # file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    console_logging_format = "%(message)s"
    file_logging_format = "%(message)s"
    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    # Add CSV logging if enabled
    if csv_logging:
        # Create CSV file path
        csv_file = os.path.join(path, file.replace('.logs', '.csv'))

        # Create CSV file if it doesn't exist
        if not os.path.isfile(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Add headers based on expected log format
                writer.writerow(['timestamp', 'epoch', 'train_loss', 'val_loss', 'val_auc', 'val_eer', 'additional_info'])

        # Create a custom handler for CSV logging
        class CSVHandler(logging.Handler):
            def emit(self, record):
                try:
                    # Parse the log message to extract structured data
                    msg = self.format(record)

                    # Extract data using regex
                    # Example log format: "Epoch 1/10 | train loss: 0.1234 | val loss: 0.5678, val auc: 0.9876, val eer: 0.1234 |"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Extract epoch information
                    epoch_match = re.search(r'Epoch (\d+)/\d+', msg)
                    epoch = epoch_match.group(1) if epoch_match else ""

                    # Extract train loss
                    train_loss_match = re.search(r'train loss: ([\d\.]+)', msg)
                    train_loss = train_loss_match.group(1) if train_loss_match else ""

                    # Extract validation metrics
                    val_loss_match = re.search(r'val loss: ([\d\.]+)', msg)
                    val_loss = val_loss_match.group(1) if val_loss_match else ""

                    val_auc_match = re.search(r'val auc: ([\d\.]+)', msg)
                    val_auc = val_auc_match.group(1) if val_auc_match else ""

                    val_eer_match = re.search(r'val eer: ([\d\.]+)', msg)
                    val_eer = val_eer_match.group(1) if val_eer_match else ""

                    # Additional info (everything after the validation metrics)
                    additional_info = ""
                    if val_eer_match:
                        pos = val_eer_match.end()
                        if pos < len(msg):
                            additional_info = msg[pos:].strip()

                    # Write to CSV file
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, epoch, train_loss, val_loss, val_auc, val_eer, additional_info])

                except Exception as e:
                    print(f"Error writing to CSV: {e}")
                    self.handleError(record)

        # Add the CSV handler to the logger
        csv_handler = CSVHandler()
        csv_handler.setLevel(logging.INFO)
        csv_handler.setFormatter(formatter)
        logger.addHandler(csv_handler)

    return logger